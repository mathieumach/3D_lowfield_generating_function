import numpy as np
import cv2
from Sequences import *

def create_3D_noisy_and_clean_data(FOV, Resolution, Bandwidth, seq, TR, TE, TI, TI2, alpha, noise_factor\
                                   ,T1_3D, T2_3D, M0_3D, B1_3D, flipangle_3D, t2_star_3D, ADC_3D, c, met, ETL, phi):
    """
    Function that will create two 3D simulation of low-field MRI sequences, one without noise and one with noise
    
    Inputs:
    
    //////// parameters to be chosen for runing the sequence ////////
    FOV          -> 1x3 array of the field of view
    Resolution   -> 1x3 array of the resolution
    Bandwidth    -> Bandwidth of aquisition
    seq          -> string defining the sequence to simulate; {'SE','GE','IN','Double IN','FLAIR','Dif'} 
    The seq strings correspond to these sequences; {spin echo, gradient echo, inversion recovery, double inversion recovery, FLAIR, diffusion}
    TR           -> repetition time, must be in milli-seconds (ex: 3000 for 3 seconds, 50 for 50 milli-seconds)
    TE           -> echo time, must be in milli-seconds (ex: 160 for 0.160 seconds, 50 for 50 milli-seconds)
    TI           -> inversion time, must be in milli-seconds (ex: 650 for 0.650 seconds, 50 for 50 milli-seconds)
    TI2          -> second inversion time, must be in milli-seconds (ex: 150 for 0.150 seconds, 50 for 50 milli-seconds)
    alpha        -> flip angle, between {0-90}
    noise_factor -> multiplying noise factor
    c            -> the number of points in the time signal generated in TSE seq (10 works well and is not to slow in terms of computation)
    met          -> is a string specifying the kspace trajectory in TSE; {'Out-in','In-out','Linear'}
    ETl          -> Echo train length
    phi          -> map, create from B0 map, representing of the difference in frequencies from center frequency

    //////// low-field systems and physiological maps ////////
    T1_3D        -> T1 relaxation map
    T2_3D        -> T2 relaxation map
    M0_3D        -> Proton density map
    B1_3D        -> B1 map
    flipangle_3D -> flipangle tensor map
    t2_star_3D   -> t2 star tensor map
    ADC_3D       -> apparent diffusion coefficient (ADC) tensor map
    
    Outputs:
    clean_data -> the simulated tensor sequence without noise
    noisy_data -> the simulated tensor sequence with noise
    
    """
    
    # Final data matrix size (field of view / reolution) 
    Data_mat = np.divide(FOV, Resolution)
    Data_mat = [int(Data_mat[0]), int(Data_mat[1]), int(Data_mat[2])]
    
    # Divide by 1000 to have the values in seconds
    TR = np.divide(TR,1000) 
    TE = np.divide(TE,1000)
    TI = np.divide(TI,1000)
    TI2 = np.divide(TI2,1000)
    
    # Computing the 3D sequence and resizing
    if seq == 'SE':
        Data_3D = spin_echo_seq(TR, TE, T1_3D, T2_3D, M0_3D)
    elif seq == 'GE':
        angle = flipangle_3D/alpha
        Data_3D =  Gradient_seq(TR, TE, T1_3D, t2_star_3D, M0_3D, angle);
    elif seq == 'IN':
        Data_3D = IN_seq(TR, TE, TI, T1_3D, T2_3D, M0_3D)
    elif seq == 'Double IN':
        Data_3D = DoubleInversion_seq(TR, TE, TI, TI2, T1_3D, T2_3D, M0_3D)
    elif seq == 'FLAIR':
        TI = np.log(2) * 3.695 
        Data_3D = IN_seq(TR, TE, TI, T1_3D, T2_3D, M0_3D)
    elif seq == 'Dif':
        Data_3D = Diffusion_seq(TR, TE, T1_3D, T2_3D, M0_3D, b, ADC_3D)
    elif seq == 'TSE':
        Data_3D = TSE_seq(TR, TE, ETL, M0_3D, T1_3D, T2_3D, c, met)
    elif seq == 'SSFP':
        Data_3D = SSFP_Echo_seq(T1_3D, T2_3D, M0_3D, alpha, phi)

    # Multiplying the data by B1 map
    Data_3D = np.multiply(Data_3D, B1_3D)

    # Resizing the data
    n_seq = np.zeros((T1_3D.shape[0], Data_mat[1], Data_mat[2]));  clean_data = np.zeros((Data_mat))
    for x in range(T1_3D.shape[0]):
        n_seq[x,:,:] = cv2.resize(Data_3D[x,:,:], dsize=(Data_mat[2], Data_mat[1]))
    for x in range(Data_mat[1]):
        clean_data[:,x,:] = cv2.resize(n_seq[:,x,:], dsize=(Data_mat[2], Data_mat[0]))
        
    ##### NOISE #####
    # The noise is a 3D tensor that will be added to the data
    shape = Data_mat
    S = [int(shape[0]/2),int(shape[1]/2),int(shape[2]/2)] # S is a vector with the center of the Data matrix
    length = 20 # size of the boxes used to compute the SNR
    half_length = length/2
    mean_box = [int(S[0]-half_length), int(S[0]+half_length), int(75-half_length), int(75+half_length)]
    mean = np.nanmean(clean_data[mean_box[0]:mean_box[1],mean_box[2]:mean_box[3],int(shape[2]/2)]) # Computes the mean of the signal in a box from center axial slice without noise
    tot_res = Resolution[0]*Resolution[1]
    tot_data = Data_mat[0]*Data_mat[1]

    # Equation below links the SNR with the resolution, bandwidth and FOV
    #noise_relation = 8.688078756400369 * tot_res / np.sqrt(np.divide(Bandwidth,tot_data)) # 8.688... is a constant computed from real a lowfield image
    noise_relation = 2 * tot_res / np.sqrt(np.divide(Bandwidth,tot_data))
    std = mean / noise_relation
    noise = np.abs(np.random.normal(0,std,Data_mat)) 

    # constant that multiply the amount of noise to add
    noise = noise*noise_factor

    # Adding the noise
    noisy_data = clean_data + noise

    #print('Simulated sequence: ' + seq)
    #print('Shape of the noisy_data tensor: ' + str(noisy_data.shape))
    #print('Shape of the clean_data tensor: ' + str(clean_data.shape))
    
    return np.abs(clean_data), np.abs(noisy_data)

def snr_homemade(im,s1,s2,s3,s4,m1,m2,m3,m4,max):
    """
    Function computing the SNR defined by the mean of the signal (computed from a 2D box, or area) divided by the standard deviation of the noise 
    (also computed from a 2D box, or area)
    
    Inputs:
    im     --> image from which the SNR will be computed
    s1, s2 --> top and bottum value of mean box
    s3, s4 --> left and rigth value of mean box
    m1, m2 --> top and bottum value of noise box
    m3, m4 --> left and rigth value of noise box
    max    --> intensity value to visualise the boxes (should be high compared to the values of the image)
    
    Outputs:
    snr    --> signal to noise ratio
    t      --> original image with mean and noise boxes boundaries highlighted by the 'max' value (only for visualization)
    """
    
    t = np.copy(im)
    t[s1:s2,s3] = max
    t[s1:s2,s4] = max
    t[s1,s3:s4] = max
    t[s2,s3:s4] = max
    t[m1:m2,m3] = max
    t[m1:m2,m4] = max
    t[m1,m3:m4] = max
    t[m2,m3:m4] = max
    
    m = np.mean(im[m1:m2,m3:m4])
    std = np.std(im[s1:s2,s3:s4])
    
    if std == 0:
        snr = 1000
    else:
        snr = m/std    
    return snr, t