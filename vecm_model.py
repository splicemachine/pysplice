import numpy as np
import pandas as pd



from mlflow.pyfunc import PythonModel

class VECM(PythonModel):
    """Class for VECM models
    """

    #@abstractmethod
    def __init__(self, quantile):

        self.quantile = quantile
        self.all_params = None

    def _my_lagmat(self, obj, maxlag):

        # based on https://www.statsmodels.org/stable/_modules/statsmodels/tsa/tsatools.html#lagmat
        """
        Create 2d array of lags.

        Parameters
        ----------
        x : array_like
            Data; if 2d, observation in rows and variables in columns.
        maxlag : int
            All lags from zero to maxlag are included.

        Returns
        -------
        lagmat : ndarray
            The array with lagged observations."""

        x = np.asarray(obj, dtype=np.double)

        dropidx = 0
        nobs, nvar = x.shape
        dropidx = nvar
        if maxlag >= nobs:
            # TODO: Do we need this value error?
            raise ValueError("maxlag should be < nobs")
        lm = np.zeros((nobs + maxlag, nvar * (maxlag + 1)))
        for k in range(0, int(maxlag + 1)):
            lm[maxlag - k:nobs + maxlag - k,
            nvar * (maxlag - k):nvar * (maxlag - k + 1)] = x


        startobs = 0
        stopobs = nobs
        lags = lm[startobs:stopobs, dropidx:]

        return lags

    def fit(self, exp_df, list_of_dict, quantile=None):

        quantile = self.quantile
        # This function goes through all groups and applies VECM on each one of them
        # Then we are binning anomalies into 4 categories - 0, not anomalies, and 1,2,3 - anomalies, depending on severity

        #Fit method should not be passed STATE columns, only actual data on which you want it to
        dict_of_params ={}
        total_groups = []
        #TODO find a way how to be it faster
        for mydict in list_of_dict:
            for i in mydict.keys():
                total_groups.append(i)

        df_anomalies = pd.DataFrame(np.zeros((exp_df.shape[0], exp_df.shape[1])), columns = exp_df.columns, index= exp_df.index)
        df_group_anomalies = pd.DataFrame(np.zeros((exp_df.shape[0], len(total_groups))), columns = range(len(total_groups)), index= exp_df.index)
        df_total_vecm_anomaly_score = pd.DataFrame(np.zeros((exp_df.shape[0], 2)), columns = ['VECM_total_score_raw','VECM_total_score'], index= exp_df.index)
        k=0
        total_num_of_columns = len(exp_df.columns)
        for mydict in list_of_dict:
            for i in mydict.keys():
                dict_of_params[k] = {}
                dict_of_params[k]['columns'] = mydict[i]['numeric']
                dict_of_params[k]['columns_str'] = mydict[i]['str']
                dict_of_params[k]['group_weight'] = len(mydict[i]['str'])/total_num_of_columns
                temp = exp_df.loc[:,mydict[i]['str']]

                ss_temp, params = self._vecm_modified(temp) #

                dict_of_params[k]['params'] = params
                #Here we are using function to  bin anomalies to 0 -no anomaly, 1-low anomal, 2- medium severity and 3 high severity
                anomalies_digitized, anom_bins = self._binning_anomalies(ss_temp, quantile)
                print('anom_Digit', np.shape(anomalies_digitized))
                print('ss_temp', np.shape(ss_temp))
                print(np.shape(df_anomalies.loc[:,mydict[i]['str']]))
                df_anomalies.loc[:,mydict[i]['str']] = anomalies_digitized

                #writing down boundaries for binning anomalies in future
                dict_of_params[k]['anom_bins'] = anom_bins
                dict_of_params[k]['lower_quantile'] = np.quantile(ss_temp, q=1-quantile, axis=0).tolist()
                dict_of_params[k]['higher_quantile'] = np.quantile(ss_temp, q=quantile, axis=0).tolist()

                # TODO check if this works ?
                dict_of_params[k]['sum_for_group_anomalies'] = 3 * len(dict_of_params[k]['columns_str'])
                df_group_anomalies.iloc[:,k] = np.sum(df_anomalies.loc[:,mydict[i]['str']], axis=1) / dict_of_params[k]['sum_for_group_anomalies']
                df_total_vecm_anomaly_score['VECM_total_score_raw'] += df_group_anomalies.iloc[:,k] * dict_of_params[k]['group_weight']
                k+=1

        df_total_vecm_anomaly_score['VECM_total_score'], anom_total_bins = self._binning_anomalies(df_total_vecm_anomaly_score['VECM_total_score_raw'].values, quantile, anomalies_flag=True)
        dict_of_params['total_anom'] ={}
        dict_of_params['total_anom']['anom_bins'] = anom_bins
        dict_of_params['total_anom']['higher_quantile'] = np.quantile(df_total_vecm_anomaly_score['VECM_total_score_raw'].values, q=quantile, axis=0).tolist()
        #TODO : add saving of anom_quantiles - you need them for masking stuff

        self.all_params = dict_of_params

        return df_anomalies, df_group_anomalies, df_total_vecm_anomaly_score, dict_of_params


    def _vecm_modified(self, endog):
        import statsmodels.api as sm
        ## This function fits the VECM on the data
        det_order=0
        k_ar_diff=1
        # modified from https://github.com/statsmodels/statsmodels/blob/main/statsmodels/tsa/vector_ar/vecm.py

        def detrend(y, order):
            if order == -1:
                return y
            result = sm.OLS(y, np.vander(np.linspace(-1, 1, len(y)),
                                    order+1)).fit()
            return result.resid, result.params.tolist()

        def resid(y, x):
            if x.size == 0:
                return y
            #TODO: Do I need to save this one too?
            #DONE: YES
            r = y - np.dot(x, np.dot(np.linalg.pinv(x), y))
            return r, np.dot(np.linalg.pinv(x), y)

        endog = np.asarray(endog)
        nobs, neqs = endog.shape

        # why this?  f is detrend transformed series, det_order is detrend data
        if det_order > -1:
            f = 0
        else:
            f = det_order

        endog, params_1 = detrend(endog, det_order)
        dx = np.diff(endog, 1, axis=0)
        z = self._my_lagmat(dx, k_ar_diff)
        z = z[k_ar_diff:]
        z, params_2 = detrend(z, f)

        dx = dx[k_ar_diff:]
        dx, params_3 = detrend(dx, f)
        r0t, param_4 = resid(dx, z)

        my_resids = np.empty(np.shape(endog))
        for i in range(0,np.shape(r0t)[1]):
            ss = r0t[:,i].tolist()
            ss.insert(0,0)
            ss.insert(0,0)
            my_resids[:,i] = ss

        return my_resids, (params_1,params_2,params_3,param_4)

    def _binning_anomalies(self, ss_temp, quantile, anomalies_flag=False):
        import numpy.ma as ma
        #Here we are using masked array to filter out only anomalous readings, calculate z_score on them and out into bins
        # Anomaly flag is needed if we are bining final anomalies' scores
        if anomalies_flag:
            ss_temp = np.reshape(ss_temp,(len(ss_temp),1))
            mask = ~(ss_temp >= np.quantile(ss_temp, q=quantile, axis=0))
        else:
            mask = ~((ss_temp >=np.quantile(ss_temp, q=quantile, axis=0)) | (ss_temp <=np.quantile(ss_temp, q=1-quantile, axis=0)))

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

            #TODO: this is completely voluntaristic selection of thresholds for bins
            #TODO: THINK OF A MORE MATH-proved way to do it
            if anomalies_flag:
                bins = np.array([m_min-1, l_min+(l_max-l_min)*0.1, l_min+(l_max-l_min)*0.3, m_max+1])
            else:
                bins = np.array([m_min-1, l_min+(l_max-l_min)*0.3, l_min+(l_max-l_min)*0.6, m_max+1])
            anomalies_digitized[:,i] = np.digitize(ss_temp_filled[:,i], bins)
            collect_bins.append(bins)

        return anomalies_digitized, collect_bins

    def _apply_vecm(self, params, endog):
        det_order=0
        k_ar_diff=1

        def detrend_apply(y, order, params_1):
            if order == -1:
                return y
            resid = y - params_1 * np.vander(np.linspace(-1, 1, len(y)),order+1)

            return resid

        def resid(y, x, params_4):
            if x.size == 0:
                return y
            r = y - np.dot(x, params_4)
            return r
        endog = np.asarray(endog)
        nobs, neqs = endog.shape

        # why this?  f is detrend transformed series, det_order is detrend data
        if det_order > -1:
            f = 0
        else:
            f = det_order

        endog = detrend_apply(endog, det_order, params[0])
        dx = np.diff(endog, 1, axis=0)
        z = self._my_lagmat(dx, k_ar_diff)
        z = z[k_ar_diff:]
        z = detrend_apply(z, f, params[1])
        dx = dx[k_ar_diff:]
        dx= detrend_apply(dx, f, params[2])
        r0t = resid(dx, z, params[3])
        my_resids = np.empty(np.shape(endog))
        for i in range(0,np.shape(r0t)[1]):
            ss = r0t[:,i].tolist()
            ss.insert(0,0)
            ss.insert(0,0)
            my_resids[:,i] = ss

        return my_resids

    def _applying_bin_anomalies(self,ss_temp, states,  dict_of_params, anomalies_flag=False):
        import numpy.ma as ma
        #Here we are using masked array to filter out only anomalous readings, calculate z_score on them and out into bins
        # Anomaly flag is needed if we are bining final anomalies' scores

        states = np.asarray(states)
        if anomalies_flag:
            anom_bins = dict_of_params['anom_bins']
            h_quant = dict_of_params['higher_quantile']
            ss_temp = np.reshape(ss_temp,(len(ss_temp),1))
            mask = ~(ss_temp >= h_quant)
        else:
            anom_bins = dict_of_params['anom_bins']
            l_quant = dict_of_params['lower_quantile']
            h_quant = dict_of_params['higher_quantile']
            mask = ~(((ss_temp >=h_quant) | (ss_temp <=l_quant)) & (states !='T'))

        anomalies_digitized = np.zeros( (np.shape(ss_temp)[0],np.shape(ss_temp)[1]) )
        ss_temp_masked = ma.masked_array(ss_temp, mask=mask)

        #Filling with a really far off value of stds, in order to have correct represenation of groups
        m_min = anom_bins[0][0] +1
        ss_temp_filled = ss_temp_masked.filled(fill_value=m_min-9000)
        for i in range(np.shape(ss_temp_masked)[1]):
            anomalies_digitized[:,i] = np.digitize(ss_temp_filled[:,i], anom_bins[i])

        return anomalies_digitized

    def apply(self, exp_df, quantile=None):
        '''This function takes in :
                exp_df : columns to which we are applying anomaly detection
                states : states for for the same columns, 'T' state means - Timeout and will be filtered
        '''
        quantile=self.quantile
        all_info = self.all_params
        value_cols = [x for x in exp_df.columns if 'STATE' not in x]
        state_cols = [x for x in exp_df.columns if 'STATE' in x]

        total_num_groups = len(all_info)-1
        df_anomalies = pd.DataFrame(np.zeros((exp_df.shape[0], len(value_cols))), columns = value_cols, index= exp_df.index)
        df_group_anomalies = pd.DataFrame(np.zeros((exp_df.shape[0], total_num_groups)), columns = list(all_info.keys())[:-1], index= exp_df.index)
        df_total_vecm_anomaly_score = pd.DataFrame(np.zeros((exp_df.shape[0], 2)), columns = ['VECM_total_score_raw','VECM_total_score'], index= exp_df.index)
        total_num_of_columns = len(exp_df.columns)
        for k in list(all_info.keys())[:-1]:
            columns_values = all_info[k]['columns_str']
            group_weight = all_info[k]['group_weight']
            temp = exp_df.loc[:,columns_values]
            params = all_info[k]['params']

            #Columns for STATE values, exactly the same order as columns_values
            columns_state = [c+'_STATE' for c in columns_values]
            states = exp_df.loc[:, columns_state]

            ss_temp = self._apply_vecm(params, temp)
            #Here we are using function to  bin anomalies to 0 -no anomaly, 1-low anomal, 2- medium severity and 3 high severity
            anomalies_digitized = self._applying_bin_anomalies(ss_temp, states, all_info[k], anomalies_flag=False)
            df_anomalies.loc[:,columns_values] = anomalies_digitized
            # Remove -states.apply if you think the sensitivity is too high with it
            adjusted_sum_for_group_anomalies = all_info[k]['sum_for_group_anomalies'] - states.apply(pd.Series.value_counts, axis=1).fillna(0)['T']
            df_group_anomalies.loc[:,k] = np.sum(df_anomalies.loc[:,columns_values ], axis=1) / adjusted_sum_for_group_anomalies
            df_total_vecm_anomaly_score['VECM_total_score_raw'] += df_group_anomalies.loc[:,k] * group_weight


        df_total_vecm_anomaly_score['VECM_total_score'] = self._applying_bin_anomalies(df_total_vecm_anomaly_score['VECM_total_score_raw'].values, _, all_info['total_anom'], anomalies_flag=True)

        return df_anomalies, df_group_anomalies, df_total_vecm_anomaly_score

    def predict(self, data):
        return self.apply(data)


    @property
    def all_params_(self):
        return self.all_params
