from typing import List, Dict, Union, Tuple, Optional, Any
import requests
from splicemachine import SpliceMachineException
from os import environ as env_vars
from requests.auth import HTTPBasicAuth

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
    DEPLOYMENTS: str = "deployments"
    FEATURES: str = "features"
    FEATURE_SETS: str = "feature-sets"
    FEATURE_SET_DESCRIPTIONS: str = "feature-set-descriptions"
    DEPLOY_FEATURE_SET: str = "deploy-feature-set"
    FEATURE_VECTOR: str = "feature-vector"
    FEATURE_VECTOR_SQL: str = "feature-vector-sql"
    TRAINING_SETS: str = "training-sets"
    TRAINING_SET_FEATURES: str = "training-set-features"
    TRAINING_SET_FROM_DEPLOYMENT: str = "training-set-from-deployment"
    TRAINING_SET_FROM_VIEW: str = "training-set-from-view"
    TRAINING_VIEWS: str = "training-views"
    TRAINING_VIEW_DESCRIPTIONS: str = "training-view-descriptions"
    TRAINING_VIEW_FEATURES: str = "training-view-features"
    TRAINING_VIEW_ID: str = "training-view-id"
    SUMMARY: str = "summary"

def make_request(url: str, endpoint: str, method: str, auth: HTTPBasicAuth,
                 params: Optional[Dict[str, Any]] = None,
                 body: Union[Dict[str,Any], List[Any]] = None) -> Union[dict,List[dict]]:
    if not auth:
        raise Exception(
            "You have not logged into Feature Store director."
            " Please run fs.login_fs(username, password)"
        )
    if not url:
        raise KeyError("Uh Oh! FS_URL variable was not found... you should call 'fs.set_feature_store_url(<url>)' before doing trying again.")
    url = f'{url}/{endpoint}'
    try:
        r = RequestType.method_map[method](url, params=params, json=body, auth=auth)
    except KeyError:
        raise SpliceMachineException(f'Not a recognized HTTP method: {method}.'
                                     f'Please use one of the following: {RequestType.get_valid()}')
    try:
        r.raise_for_status()
    except requests.exceptions.HTTPError as error:
        to_print = str(error) if error.response.status_code == 500 else error.response.json()["message"]
        raise SpliceMachineException(f'{to_print}') from None
    return r.json()
    # except requests.exceptions.HTTPError as e:
    #     print(f'Error encountered: {e.response.text}')
    #     return

def _get_feature_store_url():
    """
    Get address of Feature Store Container for Kubernetes

    :return: (str) the featurestore URI
    """
    url = env_vars.get('FS_URL')
    return url

def _get_credentials():
    """
    Returns the username and password of the user if stored in env variables

    :return: str, str
    """
    return env_vars.get('SPLICE_JUPYTER_USER'), env_vars.get('SPLICE_JUPYTER_PASSWORD')
