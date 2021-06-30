from splicemachine.features.constants import PipeLanguage, PipeType
from splicemachine import SpliceMachineException
import cloudpickle
import base64

class Pipe:
    def __init__(self, *, name, description, ptype, lang, func, code, pipe_id=None, pipe_version=None, splice_ctx=None, args=None, kwargs=None, **obj_kwargs):
        self.name = name.upper()
        self.description = description
        self.ptype = ptype
        self.lang = lang
        self.func = byteify_string(func)
        self.code = code
        self.pipe_id = pipe_id
        self.pipe_version = pipe_version
        self.splice_ctx = splice_ctx
        self._args = byteify_string(args)
        self._kwargs = byteify_string(kwargs)
        args = {k.lower(): obj_kwargs[k] for k in obj_kwargs}
        self.__dict__.update(args)

    @property
    def type(self):
        return self.ptype
    @type.setter
    def type(self, value):
        self.ptype = value

    @property
    def language(self):
        return self.lang
    @language.setter
    def language(self, value):
        self.lang = value

    @property
    def function(self):
        return self.func
    @function.setter
    def function(self, value):
        self.func = value

    @property
    def args(self):
        return cloudpickle.loads(self._args) if self._args else tuple()
    @args.setter
    def args(self, value):
        self._args = cloudpickle.dumps(value)

    @property
    def kwargs(self):
        return cloudpickle.loads(self._kwargs) if self._kwargs else {}
    @kwargs.setter
    def kwargs(self, value):
        self._kwargs = cloudpickle.dumps(value)

    def set_parameters(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def set_splice_ctx(self, splice_ctx):
        self.splice_ctx = splice_ctx

    def apply(self, *args, **kwargs):
        a = []
        req = +(self.type != PipeType.source)
        if req:
            a.append(args[0])
        if args[req:]:
            a[req:] = args[req:]
        elif self.args:
            a[req:] = self.args
        a = tuple(a)

        kw = {}
        kw.update(self.kwargs or {})
        kw.update(kwargs)

        func = cloudpickle.loads(self.func)
        if self.language == PipeLanguage.pyspark:
            func.__globals__['splice'] = self.splice_ctx
            func.__globals__['spark'] = self.splice_ctx.spark_session
        r = func(*a, **kw)
        if self.language == PipeLanguage.sql:
            if not self.splice_ctx:
                raise SpliceMachineException(f'Cannot execute SQL functions without a splice context')
            return self.splice_ctx.df(r)
        else: # 'python'
            return r

    def _to_json(self):
        return {
            'name': self.name,
            'description': self.description,
            'ptype': self.ptype,
            'lang': self.lang,
            'func': stringify_bytes(self.func),
            'code': self.code,
            'pipe_id': self.pipe_id,
            'pipe_version': self.pipe_version,
            'args': stringify_bytes(self._args),
            'kwargs': stringify_bytes(self._kwargs)
        }

    def _pandas_to_spark(self, df):
        return self.splice_ctx.pandasToSpark(df)

    def _df_to_sql(self, df):
        return self.splice_ctx.createDataFrameSql(df)

def stringify_bytes(b: bytes) -> str:
    if b is None:
        return None
    return base64.encodebytes(b).decode('ascii').strip()

def byteify_string(s: str) -> bytes:
    if s is None:
        return None
    return base64.decodebytes(s.strip().encode('ascii'))
