from typing import List, Dict, Union, Tuple
import requests
from splicemachine import SpliceMachineException
from os import environ as env_vars

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

class Endpoints:
    """
    Enum for Feature Store Endpoints
    """
    FEATURES: str = "features"
    FEATURE_SETS: str = "feature-sets"
    FEATURE_SET_DESCRIPTIONS: str = "feature-set-descriptions"
    DEPLOY_FEATURE_SET: str = "deploy-feature-set"
    FEATURE_VECTOR: str = "feature-vector"
    FEATURE_VECTOR_SQL: str = "feature-vector-sql"
    TRAINING_SETS: str = "training-sets"
    TRAINING_SET_FROM_DEPLOYMENT: str = "training-set-from-deployment"
    TRAINING_SET_FROM_VIEW: str = "training-set-from-view"
    TRAINING_VIEWS: str = "training-views"
    TRAINING_VIEW_DESCRIPTIONS: str = "training-view-descriptions"
    TRAINING_VIEW_FEATURES: str = "training-view-features"
    TRAINING_VIEW_ID: str = "training-view-id"

def make_request(url: str, endpoint: str, method: RequestType, query: Dict[str, Union[str, List[str]]] = None, body: Dict[str, str] = None) -> requests.Response:
    if not url:
        raise KeyError("Uh Oh! FS_URL variable was not found... you should call 'fs.set_feature_store_url(<url>)' before doing trying again.")
    url = f'{url}/{endpoint}'
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
        to_print = str(error) if error.response.status_code == 500 else error.response.json()["detail"]
        raise SpliceMachineException(f'{to_print}') from None
    return r
    # except requests.exceptions.HTTPError as e:
    #     print(f'Error encountered: {e.response.text}')
    #     return

def get_feature_store_url():
    """
    Get address of Feature Store Container for Kubernetes

    :return: (str) the featurestore URI
    """
    url = env_vars.get('FS_URL')
    return url