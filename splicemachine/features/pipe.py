from splicemachine import SpliceMachineException
import cloudpickle
import base64

class Pipe:
    def __init__(self, *, name, description, ptype, lang, func, code, pipe_id=None, pipe_version=None, splice_ctx=None, **kwargs):
        self.name = name.upper()
        self.description = description
        self.ptype = ptype
        self.lang = lang
        self.func = base64.decodebytes(func.encode('ascii'))
        self.code = code
        self.pipe_id = pipe_id
        self.pipe_version = pipe_version
        self.splice_ctx = splice_ctx
        args = {k.lower(): kwargs[k] for k in kwargs}
        self.__dict__.update(args)

    @property
    def type(self):
        return self.ptype
    @type.setter
    def x(self, value):
        self.ptype = value

    @property
    def language(self):
        return self.lang
    @language.setter
    def x(self, value):
        self.lang = value

    @property
    def function(self):
        return self.func
    @function.setter
    def x(self, value):
        self.func = value

    def apply(self, *args, **kwargs):   
        func = cloudpickle.loads(self.func)
        if self.language == 'pyspark':
            func.__globals__['splice'] = self.splice_ctx 
        r = func(*args, **kwargs)
        if self.language == 'sql':
            if not self.splice_ctx:
                raise SpliceMachineException(f'Cannot execute SQL functions without a splice context')
            return self.splice_ctx.df(r)
        else: # 'python'
            return r


