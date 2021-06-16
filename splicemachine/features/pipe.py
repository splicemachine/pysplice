from splicemachine import SpliceMachineException
import cloudpickle
import base64

class Pipe:
    def __init__(self, *, name, description, type, language, function, code, pipe_id=None, pipe_version=None, splice_ctx=None, **kwargs):
        self.name = name.upper()
        self.description = description
        self.type = type
        self.language = language
        self.function = base64.decodebytes(function.encode('ascii'))
        self.code = code
        self.pipe_id = pipe_id
        self.pipe_version = pipe_version
        self.splice_ctx = splice_ctx
        args = {k.lower(): kwargs[k] for k in kwargs}
        self.__dict__.update(args)

    def apply(self, *args, **kwargs):   
        func = cloudpickle.loads(self.function)
        if self.language == 'pyspark':
            func.__globals__['splice'] = self.splice_ctx 
        r = func(*args, **kwargs)
        if self.language == 'sql':
            if not self.splice_ctx:
                raise SpliceMachineException(f'Cannot execute SQL functions without a splice context')
            return self.splice_ctx.df(r)
        else: # 'python'
            return r


