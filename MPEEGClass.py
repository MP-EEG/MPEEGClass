"""
EEG DataREader Analysis
Kaczmarek's Lab  2021 at Yale University

Description:
    It is Class Pyhton to Analysis the EEG files 4 Channel

Data are located in PORT C

@author: Maysam

"""
#%% Declaration
import numpy as np
from mne.io import read_raw_edf
# from mne.datasets import eegbci
# from mne.decoding import CSP
# from mne.minimum_norm import read_inverse_operator, compute_source_psd
from mne.time_frequency import psd_welch
import glob
import mne
from mne.time_frequency import tfr_morlet, psd_multitaper, psd_welch
import matplotlib.pyplot as plt

class NeuroEEG():
    def __init__(self,raw_fnames, Infostr="info.txt",ChName=['C:F3', 'C:P3', 'C:F4', 'C:P4', 'C:Ref'],Duration=300):
        self.prmStruct=0
        self.strParam = open(Infostr, "r").read()
        self.strParam='self.prmStruct='+self.strParam
        exec(self.strParam)
        self.duration=self.prmStruct['duration']
        self.iter_freqs = self.prmStruct['iter_freqs']
        self.iter_freqs.append(  ('Raw',self.iter_freqs[0][1],self.iter_freqs[-1][-1])  )
        # Read EDF File
        self.raw_data = read_raw_edf(raw_fnames, preload=True)
        self.raw_data.filter(0,self.iter_freqs[-1][-1], fir_design='firwin', skip_by_annotation='edge')
        self.raw_data.pick_channels(ChName);
        self.X_data=[]
        for (band,fmin,fmax) in self.iter_freqs:
            A=self.raw_data.copy()
            A.filter(fmin,fmax, fir_design='firwin', skip_by_annotation='edge')
            self.X_data.append(A)   
        
        self.DeltaW=self.X_data[0];self.ThetaW=self.X_data[1];self.AlphaW=self.X_data[2];
        self.BetaW=self.X_data[3];self.GammaW=self.X_data[4];
        self.Update(Duration=Duration);
                
        
        
        
    def Update(self,Duration=300):
        self.Ts=self.DeltaW.times[1]-self.DeltaW.times[0]
        if (Duration >= self.raw_data.times[-1]):
            Dur=self.raw_data.times[-1];
        else:
            Dur=Duration;
        self.nRange_start=np.arange(60,Dur,self.duration)
        self.nRange_stop=np.roll(self.nRange_start,-1)
        self.nRange_start=self.nRange_start[0:-1]
        self.nRange_stop=self.nRange_stop[0:-1]
        self.Data_struct_ch1_prm={
            "DeltaPowerRatio":0,
            "ThetaPowerRatio":0,
            "AlphaPowerRatio":0,
            "BetaPowerRatio":0,
            "GammaPowerRatio":0,
            "TotalPower":0,
            "MedianFreq":0,
            "DeltaMeanValue":0,
            "ThetaMeanValue":0,
            "AlphaMeanValue":0,
            "BetaMeanValue":0,
            "GammaMeanValue":0,
            "DeltaVarValue":0,
            "ThetaVarValue":0,
            "AlphaVarValue":0,
            "BetaVarValue":0,
            "GammaVarValue":0,    
            }
        self.Data_struct_ch2_prm={
            "DeltaPowerRatio":0,
            "ThetaPowerRatio":0,
            "AlphaPowerRatio":0,
            "BetaPowerRatio":0,
            "GammaPowerRatio":0,
            "TotalPower":0,
            "MedianFreq":0,
            "DeltaMeanValue":0,
            "ThetaMeanValue":0,
            "AlphaMeanValue":0,
            "BetaMeanValue":0,
            "GammaMeanValue":0,
            "DeltaVarValue":0,
            "ThetaVarValue":0,
            "AlphaVarValue":0,
            "BetaVarValue":0,
            "GammaVarValue":0,    
            }
        self.Data_struct_ch3_prm={
            "DeltaPowerRatio":0,
            "ThetaPowerRatio":0,
            "AlphaPowerRatio":0,
            "BetaPowerRatio":0,
            "GammaPowerRatio":0,
            "TotalPower":0,
            "MedianFreq":0,
            "DeltaMeanValue":0,
            "ThetaMeanValue":0,
            "AlphaMeanValue":0,
            "BetaMeanValue":0,
            "GammaMeanValue":0,
            "DeltaVarValue":0,
            "ThetaVarValue":0,
            "AlphaVarValue":0,
            "BetaVarValue":0,
            "GammaVarValue":0,    
            }
        self.Data_struct_ch4_prm={
            "DeltaPowerRatio":0,
            "ThetaPowerRatio":0,
            "AlphaPowerRatio":0,
            "BetaPowerRatio":0,
            "GammaPowerRatio":0,
            "TotalPower":0,
            "MedianFreq":0,
            "DeltaMeanValue":0,
            "ThetaMeanValue":0,
            "AlphaMeanValue":0,
            "BetaMeanValue":0,
            "GammaMeanValue":0,
            "DeltaVarValue":0,
            "ThetaVarValue":0,
            "AlphaVarValue":0,
            "BetaVarValue":0,
            "GammaVarValue":0,    
            }
        
        self.Data_struct_ch1={'Delta':[],'Theta':[],'Alpha':[],'Beta':[],'Gamma':[],'Raw':[],'Freq':[],
                         'DeltaW':[],'ThetaW':[],'AlphaW':[],'BetaW':[],'GammaW':[]}
        self.Data_struct_ch2={'Delta':[],'Theta':[],'Alpha':[],'Beta':[],'Gamma':[],'Raw':[],'Freq':[],
                         'DeltaW':[],'ThetaW':[],'AlphaW':[],'BetaW':[],'GammaW':[]}
        self.Data_struct_ch3={'Delta':[],'Theta':[],'Alpha':[],'Beta':[],'Gamma':[],'Raw':[],'Freq':[],
                         'DeltaW':[],'ThetaW':[],'AlphaW':[],'BetaW':[],'GammaW':[]}
        self.Data_struct_ch4={'Delta':[],'Theta':[],'Alpha':[],'Beta':[],'Gamma':[],'Raw':[],'Freq':[],
                         'DeltaW':[],'ThetaW':[],'AlphaW':[],'BetaW':[],'GammaW':[]}
        
        self.Data_struct_ch1_arr=[];self.Data_struct_ch2_arr=[];self.Data_struct_ch3_arr=[];self.Data_struct_ch4_arr=[]
        self.Data_struct_ch1_prm_arr=[];self.Data_struct_ch2_prm_arr=[];self.Data_struct_ch3_prm_arr=[];self.Data_struct_ch4_prm_arr=[];
        
        for tmin,tmax in zip(self.nRange_start,self.nRange_stop):
            for (band,fmin,fmax) in self.iter_freqs:
                self.nfft=np.arange(0,self.duration,self.Ts).size+1
                self.psds_r, self.freqs_r = psd_welch(self.raw_data.copy(), tmin=tmin, tmax=tmax, fmin=fmin,
                                            fmax=fmax,  n_fft=self.nfft)
                self.Data_struct_ch1[band]=self.psds_r[0];self.Data_struct_ch2[band]=self.psds_r[1];
                self.Data_struct_ch3[band]=self.psds_r[2];self.Data_struct_ch4[band]=self.psds_r[3];
                self.Data_struct_ch1['Freq']=self.freqs_r;self.Data_struct_ch2['Freq']=self.freqs_r;
                self.Data_struct_ch3['Freq']=self.freqs_r;self.Data_struct_ch4['Freq']=self.freqs_r;
            self.Data_struct_ch1['DeltaW']=self.DeltaW.copy().crop(tmin=tmin,tmax=tmax).get_data()[0]
            self.Data_struct_ch2['DeltaW']=self.DeltaW.copy().crop(tmin=tmin,tmax=tmax).get_data()[1]
            self.Data_struct_ch3['DeltaW']=self.DeltaW.copy().crop(tmin=tmin,tmax=tmax).get_data()[2]
            self.Data_struct_ch4['DeltaW']=self.DeltaW.copy().crop(tmin=tmin,tmax=tmax).get_data()[3]
            
        
            self.Data_struct_ch1['ThetaW']=self.ThetaW.copy().crop(tmin=tmin,tmax=tmax).get_data()[0]
            self.Data_struct_ch2['ThetaW']=self.ThetaW.copy().crop(tmin=tmin,tmax=tmax).get_data()[1]
            self.Data_struct_ch3['ThetaW']=self.ThetaW.copy().crop(tmin=tmin,tmax=tmax).get_data()[2]
            self.Data_struct_ch4['ThetaW']=self.ThetaW.copy().crop(tmin=tmin,tmax=tmax).get_data()[3]
        
            self.Data_struct_ch1['AlphaW']=self.AlphaW.copy().crop(tmin=tmin,tmax=tmax).get_data()[0]
            self.Data_struct_ch2['AlphaW']=self.AlphaW.copy().crop(tmin=tmin,tmax=tmax).get_data()[1]
            self.Data_struct_ch3['AlphaW']=self.AlphaW.copy().crop(tmin=tmin,tmax=tmax).get_data()[2]
            self.Data_struct_ch4['AlphaW']=self.AlphaW.copy().crop(tmin=tmin,tmax=tmax).get_data()[3]
            
            self.Data_struct_ch1['BetaW']=self.BetaW.copy().crop(tmin=tmin,tmax=tmax).get_data()[0]
            self.Data_struct_ch2['BetaW']=self.BetaW.copy().crop(tmin=tmin,tmax=tmax).get_data()[1]
            self.Data_struct_ch3['BetaW']=self.BetaW.copy().crop(tmin=tmin,tmax=tmax).get_data()[2]
            self.Data_struct_ch4['BetaW']=self.BetaW.copy().crop(tmin=tmin,tmax=tmax).get_data()[3]
            
            self.Data_struct_ch1['GammaW']=self.GammaW.copy().crop(tmin=tmin,tmax=tmax).get_data()[0]
            self.Data_struct_ch2['GammaW']=self.GammaW.copy().crop(tmin=tmin,tmax=tmax).get_data()[1]
            self.Data_struct_ch3['GammaW']=self.GammaW.copy().crop(tmin=tmin,tmax=tmax).get_data()[2]
            self.Data_struct_ch4['GammaW']=self.GammaW.copy().crop(tmin=tmin,tmax=tmax).get_data()[3]
            
            self.Data_struct_ch1_arr.append(self.Data_struct_ch1.copy())
            self.Data_struct_ch2_arr.append(self.Data_struct_ch2.copy())
            self.Data_struct_ch3_arr.append(self.Data_struct_ch3.copy())
            self.Data_struct_ch4_arr.append(self.Data_struct_ch4.copy())
        #%% Real CalculationPSD
        for i in range(0,len(self.Data_struct_ch1_arr)):
            self.Data_struct_ch1=self.Data_struct_ch1_arr[i].copy()
            self.Data_struct_ch2=self.Data_struct_ch2_arr[i].copy()
            self.Data_struct_ch3=self.Data_struct_ch3_arr[i].copy()
            self.Data_struct_ch4=self.Data_struct_ch4_arr[i].copy()
            
            self.psds_r_ch1=self.Data_struct_ch1['Raw'].copy();self.psds_d_ch1=self.Data_struct_ch1['Delta'].copy()
            self.psds_t_ch1=self.Data_struct_ch1['Theta'].copy();self.psds_b_ch1=self.Data_struct_ch1['Beta'].copy()
            self.psds_g_ch1=self.Data_struct_ch1['Gamma'].copy();self.Freq_ch1=self.Data_struct_ch1['Freq'].copy()
            self.psds_a_ch1=self.Data_struct_ch1['Alpha'].copy()
            
            self.psds_r_ch2=self.Data_struct_ch2['Raw'].copy();self.psds_d_ch2=self.Data_struct_ch2['Delta'].copy()
            self.psds_t_ch2=self.Data_struct_ch2['Theta'].copy();self.psds_b_ch2=self.Data_struct_ch2['Beta'].copy()
            self.psds_g_ch2=self.Data_struct_ch2['Gamma'].copy();self.Freq_ch2=self.Data_struct_ch2['Freq'].copy()
            self.psds_a_ch2=self.Data_struct_ch2['Alpha'].copy()
            
            self.psds_r_ch3=self.Data_struct_ch3['Raw'].copy();self.psds_d_ch3=self.Data_struct_ch3['Delta'].copy()
            self.psds_t_ch3=self.Data_struct_ch3['Theta'].copy();self.psds_b_ch3=self.Data_struct_ch3['Beta'].copy()
            self.psds_g_ch3=self.Data_struct_ch3['Gamma'].copy();self.Freq_ch3=self.Data_struct_ch3['Freq'].copy()
            self.psds_a_ch3=self.Data_struct_ch3['Alpha'].copy()
            
            self.psds_r_ch4=self.Data_struct_ch2['Raw'].copy();self.psds_d_ch4=self.Data_struct_ch4['Delta'].copy()
            self.psds_t_ch4=self.Data_struct_ch2['Theta'].copy();self.psds_b_ch4=self.Data_struct_ch4['Beta'].copy()
            self.psds_g_ch4=self.Data_struct_ch2['Gamma'].copy();self.Freq_ch4=self.Data_struct_ch4['Freq'].copy()
            self.psds_a_ch4=self.Data_struct_ch2['Alpha'].copy()
            
            self.power_psds_d_ch1=sum(self.psds_d_ch1);self.power_psds_d_ch2=sum(self.psds_d_ch2)
            self.power_psds_t_ch1=sum(self.psds_t_ch1);self.power_psds_t_ch2=sum(self.psds_t_ch2)
            self.power_psds_a_ch1=sum(self.psds_a_ch1);self.power_psds_a_ch2=sum(self.psds_a_ch2)
            self.power_psds_b_ch1=sum(self.psds_b_ch1);self.power_psds_b_ch2=sum(self.psds_b_ch2)
            self.power_psds_g_ch1=sum(self.psds_g_ch1);self.power_psds_g_ch2=sum(self.psds_g_ch2)
            self.power_psds_r_ch1=sum(self.psds_r_ch1);self.power_psds_r_ch2=sum(self.psds_r_ch2)   
            
            self.power_psds_d_ch3=sum(self.psds_d_ch3);self.power_psds_d_ch4=sum(self.psds_d_ch4)
            self.power_psds_t_ch3=sum(self.psds_t_ch3);self.power_psds_t_ch4=sum(self.psds_t_ch4)
            self.power_psds_a_ch3=sum(self.psds_a_ch3);self.power_psds_a_ch4=sum(self.psds_a_ch4)
            self.power_psds_b_ch3=sum(self.psds_b_ch3);self.power_psds_b_ch4=sum(self.psds_b_ch4)
            self.power_psds_g_ch3=sum(self.psds_g_ch3);self.power_psds_g_ch4=sum(self.psds_g_ch4)
            self.power_psds_r_ch3=sum(self.psds_r_ch3);self.power_psds_r_ch4=sum(self.psds_r_ch4)   
        
            self.Data_struct_ch1_prm['DeltaPowerRatio']=round(self.power_psds_d_ch1/self.power_psds_r_ch1*100,1)
            self.Data_struct_ch2_prm['DeltaPowerRatio']=round(self.power_psds_d_ch2/self.power_psds_r_ch2*100,1)
            self.Data_struct_ch3_prm['DeltaPowerRatio']=round(self.power_psds_d_ch3/self.power_psds_r_ch3*100,1)
            self.Data_struct_ch4_prm['DeltaPowerRatio']=round(self.power_psds_d_ch4/self.power_psds_r_ch4*100,1)
            
            self.Data_struct_ch1_prm['ThetaPowerRatio']=round(self.power_psds_t_ch1/self.power_psds_r_ch1*100,1)
            self.Data_struct_ch2_prm['ThetaPowerRatio']=round(self.power_psds_t_ch2/self.power_psds_r_ch2*100,1)
            self.Data_struct_ch3_prm['ThetaPowerRatio']=round(self.power_psds_d_ch3/self.power_psds_r_ch3*100,1)
            self.Data_struct_ch4_prm['ThetaPowerRatio']=round(self.power_psds_d_ch4/self.power_psds_r_ch4*100,1)
            
            
            self.Data_struct_ch1_prm['AlphaPowerRatio']=round(self.power_psds_a_ch1/self.power_psds_r_ch1*100,1)
            self.Data_struct_ch2_prm['AlphaPowerRatio']=round(self.power_psds_a_ch2/self.power_psds_r_ch2*100,1)
            self.Data_struct_ch3_prm['AlphaPowerRatio']=round(self.power_psds_d_ch3/self.power_psds_r_ch3*100,1)
            self.Data_struct_ch4_prm['AlphaPowerRatio']=round(self.power_psds_d_ch4/self.power_psds_r_ch4*100,1)
            
            self.Data_struct_ch1_prm['BetaPowerRatio']=round(self.power_psds_b_ch1/self.power_psds_r_ch1*100,1)
            self.Data_struct_ch2_prm['BetaPowerRatio']=round(self.power_psds_b_ch2/self.power_psds_r_ch2*100,1)
            self.Data_struct_ch3_prm['BetaPowerRatio']=round(self.power_psds_d_ch3/self.power_psds_r_ch3*100,1)
            self.Data_struct_ch4_prm['BetaPowerRatio']=round(self.power_psds_d_ch4/self.power_psds_r_ch4*100,1)
        
            self.Data_struct_ch1_prm['GammaPowerRatio']=round(self.power_psds_g_ch1/self.power_psds_r_ch1*100,1)
            self.Data_struct_ch2_prm['GammaPowerRatio']=round(self.power_psds_g_ch2/self.power_psds_r_ch2*100,1)
            self.Data_struct_ch3_prm['GammaPowerRatio']=round(self.power_psds_d_ch3/self.power_psds_r_ch3*100,1)
            self.Data_struct_ch4_prm['GammaPowerRatio']=round(self.power_psds_d_ch4/self.power_psds_r_ch4*100,1)
        
            self.Data_struct_ch1_prm['TotalPower']=self.power_psds_r_ch1
            self.Data_struct_ch2_prm['TotalPower']=self.power_psds_r_ch2
            self.Data_struct_ch3_prm['TotalPower']=self.power_psds_r_ch3
            self.Data_struct_ch4_prm['TotalPower']=self.power_psds_r_ch4
        
            self.Data_struct_ch1_prm['DeltaMeanValue']=self.Data_struct_ch1['DeltaW'].copy().mean()
            self.Data_struct_ch2_prm['DeltaMeanValue']=self.Data_struct_ch2['DeltaW'].copy().mean()
            self.Data_struct_ch3_prm['DeltaMeanValue']=self.Data_struct_ch3['DeltaW'].copy().mean()
            self.Data_struct_ch4_prm['DeltaMeanValue']=self.Data_struct_ch4['DeltaW'].copy().mean()
        
            self.Data_struct_ch1_prm['ThetaMeanValue']=self.Data_struct_ch1['ThetaW'].copy().mean()
            self.Data_struct_ch2_prm['ThetaMeanValue']=self.Data_struct_ch2['ThetaW'].copy().mean()
            self.Data_struct_ch3_prm['ThetaMeanValue']=self.Data_struct_ch3['ThetaW'].copy().mean()
            self.Data_struct_ch4_prm['ThetaMeanValue']=self.Data_struct_ch4['ThetaW'].copy().mean()
        
            self.Data_struct_ch1_prm['AlphaMeanValue']=self.Data_struct_ch1['AlphaW'].copy().mean()
            self.Data_struct_ch2_prm['AlphaMeanValue']=self.Data_struct_ch2['AlphaW'].copy().mean()
            self.Data_struct_ch3_prm['AlphaMeanValue']=self.Data_struct_ch3['AlphaW'].copy().mean()
            self.Data_struct_ch4_prm['AlphaMeanValue']=self.Data_struct_ch4['AlphaW'].copy().mean()
        
            self.Data_struct_ch1_prm['BetaMeanValue']=self.Data_struct_ch1['BetaW'].copy().mean()
            self.Data_struct_ch2_prm['BetaMeanValue']=self.Data_struct_ch2['BetaW'].copy().mean()
            self.Data_struct_ch3_prm['BetaMeanValue']=self.Data_struct_ch3['BetaW'].copy().mean()
            self.Data_struct_ch4_prm['BetaMeanValue']=self.Data_struct_ch4['BetaW'].copy().mean()
        
            self.Data_struct_ch1_prm['GammaMeanValue']=self.Data_struct_ch1['GammaW'].copy().mean()
            self.Data_struct_ch2_prm['GammaMeanValue']=self.Data_struct_ch2['GammaW'].copy().mean()
            self.Data_struct_ch3_prm['GammaMeanValue']=self.Data_struct_ch3['GammaW'].copy().mean()
            self.Data_struct_ch4_prm['GammaMeanValue']=self.Data_struct_ch4['GammaW'].copy().mean()
        
            self.Data_struct_ch1_prm['DeltaVarValue']=self.Data_struct_ch1['DeltaW'].copy().var()
            self.Data_struct_ch2_prm['DeltaVarValue']=self.Data_struct_ch2['DeltaW'].copy().var()
            self.Data_struct_ch3_prm['DeltaVarValue']=self.Data_struct_ch3['DeltaW'].copy().var()
            self.Data_struct_ch4_prm['DeltaVarValue']=self.Data_struct_ch4['DeltaW'].copy().var()
        
            self.Data_struct_ch1_prm['ThetaVarValue']=self.Data_struct_ch1['ThetaW'].copy().var()
            self.Data_struct_ch2_prm['ThetaVarValue']=self.Data_struct_ch2['ThetaW'].copy().var()
            self.Data_struct_ch3_prm['ThetaVarValue']=self.Data_struct_ch3['ThetaW'].copy().var()
            self.Data_struct_ch4_prm['ThetaVarValue']=self.Data_struct_ch4['ThetaW'].copy().var()
        
            self.Data_struct_ch1_prm['AlphaVarValue']=self.Data_struct_ch1['AlphaW'].copy().var()
            self.Data_struct_ch2_prm['AlphaVarValue']=self.Data_struct_ch2['AlphaW'].copy().var()
            self.Data_struct_ch3_prm['AlphaVarValue']=self.Data_struct_ch3['AlphaW'].copy().var()
            self.Data_struct_ch4_prm['AlphaVarValue']=self.Data_struct_ch4['AlphaW'].copy().var()
        
            self.Data_struct_ch1_prm['BetaVarValue']=self.Data_struct_ch1['BetaW'].copy().var()
            self.Data_struct_ch2_prm['BetaVarValue']=self.Data_struct_ch2['BetaW'].copy().var()
            self.Data_struct_ch3_prm['BetaVarValue']=self.Data_struct_ch3['BetaW'].copy().var()
            self.Data_struct_ch4_prm['BetaVarValue']=self.Data_struct_ch4['BetaW'].copy().var()
        
            self.Data_struct_ch1_prm['GammaVarValue']=self.Data_struct_ch1['GammaW'].copy().var()
            self.Data_struct_ch2_prm['GammaVarValue']=self.Data_struct_ch2['GammaW'].copy().var()
            self.Data_struct_ch3_prm['GammaVarValue']=self.Data_struct_ch3['GammaW'].copy().var()
            self.Data_struct_ch4_prm['GammaVarValue']=self.Data_struct_ch4['GammaW'].copy().var()
        
            self.Tar1=self.power_psds_r_ch1/2;
            for k in range(0,len(self.psds_r_ch1)):
                Cr=self.psds_r_ch1[0:k]
                Cr_power=sum(Cr)
                if Cr_power>self.Tar1:
                    self.Data_struct_ch1_prm['MedianFreq']=self.Freq_ch1[k]
                    break
                    
            self.Tar2=self.power_psds_r_ch2/2;
            for k in range(0,len(self.psds_r_ch2)):
                Cr=self.psds_r_ch2[0:k]
                Cr_power=sum(Cr)
                if Cr_power>self.Tar2:
                    self.Data_struct_ch2_prm['MedianFreq']=self.Freq_ch2[k]
                    break
                      
            self.Tar3=self.power_psds_r_ch3/2;
            for k in range(0,len(self.psds_r_ch3)):
                Cr=self.psds_r_ch3[0:k]
                Cr_power=sum(Cr)
                if Cr_power>self.Tar3:
                    self.Data_struct_ch3_prm['MedianFreq']=self.Freq_ch3[k]
                    break
                      
            self.Tar4=self.power_psds_r_ch4/2;
            for k in range(0,len(self.psds_r_ch4)):
                Cr=self.psds_r_ch4[0:k]
                Cr_power=sum(Cr)
                if Cr_power>self.Tar4:
                    self.Data_struct_ch4_prm['MedianFreq']=self.Freq_ch4[k]
                    break
          
            self.Data_struct_ch1_prm_arr.append(self.Data_struct_ch1_prm.copy());
            self.Data_struct_ch2_prm_arr.append(self.Data_struct_ch2_prm.copy());
            self.Data_struct_ch3_prm_arr.append(self.Data_struct_ch3_prm.copy());
            self.Data_struct_ch4_prm_arr.append(self.Data_struct_ch4_prm.copy());
            
    def GetX_prmmat(self,ch=1):
        m=len(self.Data_struct_ch1_prm_arr)
        n=len(self.Data_struct_ch1_prm)
        X=np.zeros((m,n))
        if (ch==1):            
            for i,dt_prm_k in enumerate(self.Data_struct_ch1_prm_arr):
                X[i,:]=np.array(list(dt_prm_k.values())).reshape(1,-1)
                
        elif(ch==2):
            for i,dt_prm_k in enumerate(self.Data_struct_ch2_prm_arr):
                X[i,:]=np.array(list(dt_prm_k.values())).reshape(1,-1)
        elif(ch==3):
            for i,dt_prm_k in enumerate(self.Data_struct_ch3_prm_arr):
                X[i,:]=np.array(list(dt_prm_k.values())).reshape(1,-1)
        elif(ch==4):
            for i,dt_prm_k in enumerate(self.Data_struct_ch4_prm_arr):
                X[i,:]=np.array(list(dt_prm_k.values())).reshape(1,-1)
        return X;        

    def GetX_prmmat_ave(self,ch=1):
        m=len(self.Data_struct_ch1_prm_arr)
        n=len(self.Data_struct_ch1_prm)
        X=np.zeros((m,n))
        if (ch==1):            
            for i,dt_prm_k in enumerate(self.Data_struct_ch1_prm_arr):
                X[i,:]=np.array(list(dt_prm_k.values())).reshape(1,-1);                
        elif(ch==2):
            for i,dt_prm_k in enumerate(self.Data_struct_ch2_prm_arr):
                X[i,:]=np.array(list(dt_prm_k.values())).reshape(1,-1);
        elif(ch==3):
            for i,dt_prm_k in enumerate(self.Data_struct_ch3_prm_arr):
                X[i,:]=np.array(list(dt_prm_k.values())).reshape(1,-1);
        elif(ch==4):
            for i,dt_prm_k in enumerate(self.Data_struct_ch4_prm_arr):
                X[i,:]=np.array(list(dt_prm_k.values())).reshape(1,-1);
        X=np.average(X,axis=0).reshape((1,-1))
        return X;  

def cmpNeuroEEGplt(ctrl: NeuroEEG,drg: NeuroEEG,ch=1):
    X_ctrl_ave=ctrl.GetX_prmmat_ave(ch).ravel()
    X_drg_ave=drg.GetX_prmmat_ave(ch).ravel()
    barWidth = 0.25
    fig = plt.subplots(figsize =(8, 15))
     
    # set height of bar
    IT = X_ctrl_ave
    ECE = X_drg_ave
    
     
    # Set position of bar on X axis
    br1 = np.arange(len(IT))
    br2 = [x + barWidth for x in br1]
     
    # Make the plot
    plt.bar(br1, IT, color ='b', width = barWidth,
            edgecolor ='grey', label ='Control')
    plt.bar(br2, ECE, color ='r', width = barWidth,
            edgecolor ='grey', label ='Drug')
     
    # Adding Xticks
    plt.xlabel('Features', fontweight ='bold', fontsize = 15)
    plt.ylabel('Feature Value', fontweight ='bold', fontsize = 15)
    plt.xticks([r + barWidth for r in range(len(IT))],
            list(ctrl.Data_struct_ch1_prm.keys()))
     
    plt.legend()
    plt.show()
        
    
def NeuroEEGplt(ctrl: NeuroEEG,drg: NeuroEEG,ch=1):
    X_ctrl=ctrl.GetX_prmmat(ch)
    X_drg=drg.GetX_prmmat(ch)
    
    barWidth = 0.2
    
    for i,FeatureName in enumerate(list(ctrl.Data_struct_ch2_prm.keys())):
        # Set position of bar on X axis
        
        plt.figure()
        Ysignal_ctrl=X_ctrl[:,i].ravel();
        Ysignal_drg=X_drg[:,i].ravel();
        
        br1 = np.arange(len(Ysignal_ctrl));
        br2 = np.arange(br1[-1]+1,br1[-1]+1+ len(Ysignal_drg));
        # Make the plot
        plt.bar(br1, Ysignal_ctrl, color ='b', width = barWidth,
                edgecolor ='grey', label ='Control')
        plt.bar(br2, Ysignal_drg, color ='r', width = barWidth,
                edgecolor ='grey', label ='Drug')         
        # Adding Xticks
        plt.xlabel('Sample Elapse', fontweight ='bold', fontsize = 15)
        plt.ylabel(FeatureName, fontweight ='bold', fontsize = 15) 
        plt.title(FeatureName +'For Channel number' + str(ch), fontweight ='bold', fontsize = 15)        
        plt.legend()
        plt.show()
            