# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 15:38:23 2018

@author: victo
"""



def fish_info_feat(Features, lst_data, dset):
    for i in lst_data :
        Dset_int = dset[i]
        Col=np.apply_along_axis(lambda x : pyeeg.fisher_info(x,1,4),1, Dset_int)
        Features[str(i)+'_fish_info']=Col
    return Features


def abs_mean(Features, lst_data, dset):
    
    for i in lst_data :
        Dset_int=dset[i]
        Col = np.apply_along_axis(np.mean,1,np.apply_along_axis(abs,1,Dset_int))
        Features[str(i)+'_abs_mean']=Col
        
        
    return Features


def mean(Features, lst_data, dset):
    
     for i in lst_data :
        Dset_int=dset[i]
        Col = np.apply_along_axis(np.mean,1,Dset_int)
        Features[str(i)+'_mean']=Col
        
     return Features


def max_value(Features, lst_data, dset):
    
    for i in lst_data :
        Dset_int=dset[i]
        Col = np.apply_along_axis(max,1,Dset_int)
        Features[str(i)+'_max_value']=Col
        
    return Features

def min_value(Features, lst_data, dset) :        
    
    for i in lst_data :
        
        Dset_int=dset[i]
        Col = np.apply_along_axis(min,1,Dset_int)
        Features[str(i)+'_min_value']=Col
        
    return Features


def max_abs_value(Features, lst_data, dset):
    
    for i in lst_data :
        Dset_int=dset[i]
        Col = np.apply_along_axis(max,1,np.apply_along_axis(abs,1,Dset_int))
        Features[str(i)+'_max_abs_value']=Col
        
    return Features


#
#def derivate(Features, lst_data, dset):
#    lst_data = list(f.keys())
#    liste_derivate = []
#    for i in lst_data :
#        Dset_int=dset[i]
#        multiplicative_coef = len(Dset_int[1])/30
#        liste_derivate.append(np.apply_along_axis(np.gradient,1,Dset_int)*multiplicative_coef)
#    return liste_derivate
#
#D = derivate(Features, lst_data, dset)
#
#def abs_derivate(Features, lst_data, dset):
#    
#    lst_data = list(f.keys())
#    liste_abs_derivate = []
#    for i in lst_data :
#        Dset_int=dset[i]
#        multiplicative_coef = len(Dset_int[1])/30
#        Col = np.apply_along_axis(abs,1,np.apply_along_axis(np.gradient,1,Dset_int)*multiplicative_coef)
#        Features[str(i)+'_fish_info']=Col
#        Res_int = DF.values
#        liste_abs_derivate.append(Res_int)
#    return liste_abs_derivate
#

def abs_mean_derivate(Features, lst_data, dset):
   
    for i in lst_data :
        Dset_int=dset[i]
        multiplicative_coef = len(Dset_int[1])/30
        Col = np.apply_along_axis(np.mean,1,np.apply_along_axis(abs,1,np.apply_along_axis(np.gradient,1,Dset_int)*multiplicative_coef))
        Features[str(i)+'_abs_mean_derivate']=Col
        
    return Features


def max_abs_derivate(Features, lst_data, dset):
    
    for i in lst_data :
        Dset_int=dset[i]
        multiplicative_coef = len(Dset_int[1])/30
        Col = np.apply_along_axis(max,1,np.apply_along_axis(abs,1,np.apply_along_axis(np.gradient,1,Dset_int)*multiplicative_coef))
        Features[str(i)+'_max_abs_derivate']=Col
        
    return Features


def max_value_derivate(Features, lst_data, dset):
    
    for i in lst_data :
        Dset_int=dset[i]
        multiplicative_coef = len(Dset_int[1])/30
        Col = np.apply_along_axis(max,1,np.apply_along_axis(np.gradient,1,Dset_int)*multiplicative_coef)
        Features[str(i)+'_max_value_derivate']=Col
       
    return Features

Features = max_value_derivate(Features, lst_data, dset)

def min_value_derivate(Features, lst_data, dset):
    
    for i in lst_data :
        Dset_int=dset[i]
        multiplicative_coef = len(Dset_int[1])/30
        Col = np.apply_along_axis(min,1,np.apply_along_axis(np.gradient,1,Dset_int)*multiplicative_coef)
        Features[str(i)+'_min_value_derivate']=Col
        
    return Features

#
#def fast_fourier(Features, lst_data, dset):
#    lst_data = list(f.keys())
#    liste_abs_mean = []
#    for i in lst_data :
#        Dset_int=dset[i]
#        liste_abs_mean.append(np.apply_along_axis(np.fft.fft,1,Dset_int))
#    return liste_abs_mean
#
#FFT = fast_fourier(Features, lst_data, dset)



###########################################################
def max_amplitude_fft(Features, lst_data, dset):
    
    for i in lst_data :
        Dset_int=dset[i]
        Col = np.apply_along_axis(max,1,(np.apply_along_axis(abs,1,np.apply_along_axis(np.fft.fft,1,Dset_int))))
        Features[str(i)+'_max_amplitude_fft']=Col
       
    return Features



def freq_max_amplitude_fft(Features, lst_data, dset):
    
    
    for i in  lst_data:
        Dset_int=dset[i]
        Col = pd.DataFrame(index = np.arange(Dset_int.shape[0]))
        Col['freq'] = ""
        FFTf = np.fft.fftfreq(len(Dset_int[1]))*len(Dset_int[1])/30
        FFTf = pd.DataFrame(FFTf)
        FFTi =np.apply_along_axis(np.argmax,1,( np.apply_along_axis(abs,1,np.apply_along_axis(np.fft.fft,1,Dset_int))))
        FFTi = pd.DataFrame(FFTi)
        for j in range(Dset_int.shape[0]) : 
            Col.iloc[j,0] = FFTf.iloc[abs(FFTi.iloc[j,0]),0]

        Features[str(i)+'_freq_max_amplitude_fft'] = Col  
        
    return Features


def peak(Features, lst_data, dset):
    
    for i in  lst_data:
        
        Dset_int=dset[i]
        Tableau_signal = pd.DataFrame(Dset_int[:,:])
        Col = pd.DataFrame(index = np.arange(Dset_int.shape[0]))
        Col['peak'] = ""
        DF = np.apply_along_axis(scipy.signal.find_peaks,1,np.apply_along_axis(abs,1,Dset_int),distance = Dset_int.shape[1])
        DF = pd.DataFrame(DF)
        DF = DF[0]
        DF = DF.apply(int)
        
        for j in range(Dset_int.shape[0]) : 
            Col.iloc[j,0] = Tableau_signal.iloc[j,DF.iloc[j]]/np.mean(abs(Tableau_signal.iloc[j,:]))
        
        Features[str(i)+'_peak'] = Col   
        
    return Features
#
#Features = peak(Features, lst_data, dset)
#
#end = time.time()
#print(end - start)
#
#plt.plot([i for i in range (len(dset['eeg_1'][0,:]))],dset['eeg_1'][0,:])
#plt.show()

def puissance_moy_periodogram(Features, lst_data, dset) : 
   
     
    for i in lst_data :
        Dset_int=dset[i]
        Periodrogram_array =np.apply_along_axis(scipy.signal.periodogram,1,Dset_int)
        Periodrogram_array = Periodrogram_array[:,1,:] 
        Col = np.apply_along_axis(np.mean,1,Periodrogram_array)
        Features[str(i)+'__mean_power_periodogram']=Col
       
    return Features
    

def freq_max_power_periodogram(Features, lst_data, dset):
    
    for i in lst_data :
        Dset_int=dset[i]
        Periodrogram_array =np.apply_along_axis(scipy.signal.periodogram,1,Dset_int)
        Periodrogram_array = Periodrogram_array[:,1,:] 
        Col = np.apply_along_axis(max,1,Periodrogram_array)
        Features[str(i)+'_freq_max_power_periodogram']=Col
       
    return Features


def freq_max_value_periodogram(Features, lst_data, dset):
    
    for i in  lst_data:
        
        Dset_int=dset[i]
        Col = pd.DataFrame(index = np.arange(Dset_int.shape[0]))
        Col['freq'] = ""
        Periodrogram_array = np.apply_along_axis(scipy.signal.periodogram,1,Dset_int)
        Periodogram_frequences = Periodrogram_array[1,0,:]
        Periodogram_frequences = pd.DataFrame(Periodogram_frequences)
        Periodrogram_array = Periodrogram_array[:,1,:] 
        Periodrogram_array = np.apply_along_axis(np.argmax,1,Periodrogram_array)
        Periodrogram_array = pd.DataFrame(Periodrogram_array)
        
        for j in range(Dset_int.shape[0]) : 
            Col.iloc[j,0] = Periodogram_frequences.iloc[abs(Periodrogram_array.iloc[j,0]),0]
    
        Features[str(i)+'_freq_max_value_periodrogram'] = Col  
        
    return Features
    

def mean_amplitude_fft(Features, lst_data, dset):
    
    for i in lst_data :
        Dset_int=dset[i]
        Col = np.apply_along_axis(np.mean,1,(np.apply_along_axis(abs,1,np.apply_along_axis(np.fft.fft,1,Dset_int))))
        Features[str(i)+'_mean_amplitude_fft']=Col
       
    return Features


#Bin frequences

def bin_power(X, Band, Fs):
    
    """Compute power in each frequency bin specified by Band from FFT result of
    X. By default, X is a real signal.
    Note
    -----
    A real signal can be synthesized, thus not real.
    Parameters
    -----------
    Band
        list
        boundary frequencies (in Hz) of bins. They can be unequal bins, e.g.
        [0.5,4,7,12,30] which are delta, theta, alpha and beta respectively.
        You can also use range() function of Python to generate equal bins and
        pass the generated list to this function.
        Each element of Band is a physical frequency and shall not exceed the
        Nyquist frequency, i.e., half of sampling frequency.
     X
        list
        a 1-D real time series.
    Fs
        integer
        the sampling rate in physical frequency
    Returns
    -------
    Power
        list
        spectral power in each frequency bin.
    Power_ratio
        list
        spectral power in each frequency bin normalized by total power in ALL
        frequency bins.
    """

    C = np.fft.fft(X)
    C = abs(C)
    Power = np.zeros(len(Band) - 1)
    for Freq_Index in range(0, len(Band) - 1):
        Freq = float(Band[Freq_Index])
        Next_Freq = float(Band[Freq_Index + 1])
        Power[Freq_Index] = np.sum(
            C[int(Freq / Fs * len(X)) : int(Next_Freq / Fs * len(X))]
        )
    Power_Ratio = Power / sum(Power)
    return Power, Power_Ratio


def bin_power_features(Features, lst_data, dset):
    
    for i in lst_data :
        Dset_int=dset[i]
        Resultat_int = np.apply_along_axis(bin_power,1,Dset_int, Band = [0.5,4,7,12,30], Fs = len(Dset_int[1])/30)
        Array_somme_frequence = Resultat_int[:,0,:]
        Array_somme_frequence = pd.DataFrame(Array_somme_frequence)
        Array_somme_frequence.columns = ['Delta','Theta','Alpha','Beta']
        Array_somme_frequence_normalisee = Resultat_int[:,1,:]
        Array_somme_frequence_normalisee = pd.DataFrame(Array_somme_frequence_normalisee)
        Array_somme_frequence_normalisee.columns = ['Delta_N','Theta_N','Alpha_N','Beta_N']
        Features[str(i)+'Delta'] = Array_somme_frequence['Delta']
        Features[str(i)+'Theta'] = Array_somme_frequence['Theta']
        Features[str(i)+'Alpha'] = Array_somme_frequence['Alpha']
        Features[str(i)+'Beta'] = Array_somme_frequence['Beta']
        Features[str(i)+'Delta_N'] = Array_somme_frequence_normalisee['Delta_N']
        Features[str(i)+'Theta_N'] = Array_somme_frequence_normalisee['Theta_N']
        Features[str(i)+'Alpha_N'] = Array_somme_frequence_normalisee['Alpha_N']
        Features[str(i)+'Beta_N'] = Array_somme_frequence_normalisee['Beta_N']
       
    return Features
  
importances = model.feature_importances_
std = np.std([tree.feature_importances_ for tree in model.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

#for f in range(Features.shape[1]):
for f in range(model.n_features_):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))


# Plot the feature importances of the forest
plt.close()
plt.figure()
plt.title("Feature importances")
plt.barh(list(Features.columns), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.ylim([-1, Features.shape[1]])
plt.show()

#MMD

def max_value_sampled(Features, lst_data, dset):
    
    for i in lst_data :
        Dset_int=dset[i]
        length_sample = len(Dset_int[0])
        number_of_samples = int(length_sample/100)
        Data_Frame = pd.DataFrame()
        
        for j in range(number_of_samples):
            Sample = Dset_int[:,j*100:(j+1)*100]
            Col_M = np.apply_along_axis(max,1,Sample)
            Col_m = np.apply_along_axis(min,1,Sample)
            Col = Col_M-Col_m
            Features[str(i)+'_max_value_sample_'+str(j)]=Col
            Data_Frame['_max_value_sample_'+str(j)] = Col
        
        Features[str(i)+'MMD'+str(j)] = Data_Frame.sum(1)
    
    return Features
