from typing import List, Dict, Union, Tuple
import requests
from splicemachine import SpliceMachineException

class RequestType:
    """
    Enum for HTTP Request Types 
    """
    GET: str = "GET"
    POST: str = "POST"
    PUT: str = "PUT"
    DELETE: str = "DELETE"

    method_map = { 
        GET: requests.get,
        POST: requests.post,
        PUT: requests.put,
        DELETE: requests.delete
    }

    @staticmethod
    def get_valid() -> Tuple[str]:
        return (RequestType.GET, RequestType.POST, RequestType.PUT, RequestType.DELETE) 

def make_request(endpoint: str, method: RequestType, query: Dict[str, Union[str, List[str]]] = None, body: Dict[str, str] = None) -> requests.Response:
    url = f'http://localhost:8000/{endpoint}'
    if query:
        queries = []
        for key, value in query.items():
            if isinstance(value, list):
                queries.extend([f'{key}={v}' for v in value])
            else:
                queries.append(f'{key}={value}')
        url += '?' + '&'.join(queries)
    try:
        r = RequestType.method_map[method](url, json=body)
    except KeyError:
        raise SpliceMachineException(f'Not a recognized HTTP method: {method}.'
                                     f'Please use one of the following: {RequestType.get_valid()}')
    try:
        r.raise_for_status()
    except requests.exceptions.HTTPError as error:
        to_print = error.response.json()["detail"] if "detail" in error.response.json() else str(error)
        raise SpliceMachineException(f'{to_print}')
    return r
    # except requests.exceptions.HTTPError as e:
    #     print(f'Error encountered: {e.response.text}')
    #     return