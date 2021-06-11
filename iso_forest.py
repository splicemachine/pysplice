from splicemachine.mlflow_support import *
from h2o.estimators import H2OIsolationForestEstimator
import numpy as np
import numpy.ma as ma
import json

def IF_bin_anomalies(ss_temp, quantile):
    #Here we are using masked array to filter out only anomalous readings, calculate z_score on them and out into bins

    ss_temp = np.reshape(ss_temp,(len(ss_temp),1))
    mask = ~(ss_temp >= np.quantile(ss_temp, q=quantile, axis=0))
    quantile_values_to_save = np.quantile(ss_temp, q=quantile, axis=0)

    anomalies_digitized = np.zeros( (np.shape(ss_temp)[0],np.shape(ss_temp)[1]) )
    ss_temp_masked = ma.masked_array(ss_temp, mask=mask)

    #Filling with a really far off value of stds, in order to have correct represenation of groups
    m_min = np.min(ss_temp_masked)
    m_max = np.max(ss_temp_masked)
    ss_temp_filled = ss_temp_masked.filled(fill_value=m_min-9000)
    collect_bins = []
    for i in range(np.shape(ss_temp_masked)[1]):

        #bins are constructed based on quantiles
        l_max = np.max(ss_temp_masked[:,i])
        l_min = np.min(ss_temp_masked[:,i])

        bins = np.array([m_min-1, l_min+(l_max-l_min)*0.1, l_min+(l_max-l_min)*0.4, m_max+1])

        anomalies_digitized[:,i] = np.digitize(ss_temp_filled[:,i], bins)
        collect_bins.append(bins)

    return anomalies_digitized, collect_bins, quantile_values_to_save

def apply_IF_bin_anomalies(ss_temp, bins, quantiles_values):
    #Here we are using masked array to filter out only anomalous readings, calculate z_score on them and out into bins

    ss_temp = np.reshape(ss_temp,(len(ss_temp),1))
    mask = ~(ss_temp >= quantiles_values)

    anomalies_digitized = np.zeros( (np.shape(ss_temp)[0],np.shape(ss_temp)[1]) )
    ss_temp_masked = ma.masked_array(ss_temp, mask=mask)

    #Filling with a really far off value of stds, in order to have correct represenation of groups
    m_min = bins[0][0] + 1
    ss_temp_filled = ss_temp_masked.filled(fill_value=m_min-9000)

    for i in range(len(bins)):
        #bins are constructed based on quantiles
        anomalies_digitized[:,i] = np.digitize(ss_temp_filled[:,i], bins[i])

    return anomalies_digitized


def isolation_forest(hf,group_name,if_thr, cv_params):

    mlflow.start_run(run_name=group_name)


    model = H2OIsolationForestEstimator(sample_rate = cv_params['sample_rate'],
                                        max_depth = cv_params['max_depth'],
                                        ntrees = cv_params['ntrees'])
    mlflow.log_params(cv_params)
    with mlflow.timer('fit time', param=False):
        model.train(training_frame=hf)
    # Calculate score
    score = model.predict(hf)
    result_pred = score["predict"]
    score_df = h2o.as_list(score, use_pandas = True)
    mlflow.log_model(model, 'IFmodel'+group_name)

    #Binning anomalies - transforming the raw score into the 0-3 bins, bins should be saved,
    #because you'll need to apply the same bins for your real-time results
    score_df['if_anomaly_'+group_name], bins, quantiles = IF_bin_anomalies(score_df['predict'].values, if_thr)
    json.dump(bins, open('bins.json', 'w'), default=str)
    mlflow.log_artifact('bins.json')
    json.dump(quantiles, open('quantiles.json', 'w'), default=str)
    mlflow.log_artifact('quantiles.json')

    mlflow.end_run()


    return score_df
