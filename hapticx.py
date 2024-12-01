import numpy as np
import pandas as pd
# Visualización
import matplotlib as mpl
import matplotlib.pyplot as plt
#%matplotlib inline
import seaborn as sns
import sounddevice as sd
import pyaudio

import wave
# importing matplotlib modules
import matplotlib.image as mpimg 
import cv2

import math
#solo con el ojo derecho veo las cosas más a la izquierda
#con el ojo izq veo las cosas más a la derecha.


from scipy.fftpack import fft
#Fourier

def generate_sound_df(df_frecuencias, duration, amplitud0, sample_rate):
    """
    Genera una señal de sonido con una frecuencia dada y una fase dependiente de la posición en el DataFrame.

    Parameters:
    df_frecuencias (pd.DataFrame): DataFrame de frecuencias.
    duration (float): Duración de la señal en segundos.
    amplitud (float): Amplitud de la señal.
    sample_rate (int): Tasa de muestreo en Hz.

    Returns:
    pd.DataFrame: DataFrame con las señales de sonido generadas.
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    def funcion_signal(frequency, index, column):
        distancia = ((index+10)/10)**2 + ((column+10)/10)**2 #para evitar inconv de div por 0 , solo es una const
        phase = np.sqrt(distancia)
        amplitud = np.log(1 + amplitud0/distancia)
#        amplitud = np.log(amplitud0/distancia)
        return amplitud * np.sin(2 * np.pi * frequency * t + phase)

    signal = pd.DataFrame(index=df_frecuencias.index, columns=df_frecuencias.columns)

    for i in df_frecuencias.index:
        for j in df_frecuencias.columns:
            frequency = df_frecuencias.loc[i, j]
            signal.loc[i, j] = funcion_signal(frequency, i, j)
    
    return signal
    
    
def plot_signal(signal, sample_rate):
    """
    Muestra la señal de sonido utilizando matplotlib.

    Parameters:
    signal (numpy.ndarray): Señal de sonido.
    sample_rate (int): Tasa de muestreo en Hz.
    """
    t = np.arange(len(signal)) / sample_rate
    plt.figure(figsize=(10, 4))
    plt.plot(t, signal)
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Amplitud')
    plt.title('Señal de sonido')
    plt.show()
    

def play_sound(signal, sample_rate):
    """
    Reproduce la señal de sonido utilizando pyaudio.

    Parameters:
    signal (numpy.ndarray): Señal de sonido.
    sample_rate (int): Tasa de muestreo en Hz.
    """
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=sample_rate,
                    output=True)
    
    stream.write(signal.astype(np.float32).tobytes())
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    

def intensidad_to_freq(df, a, fi): # la frecuencia incrementa en 1 cada a valores de pixel
    #la frec inicial es fi
    df_ = df/a
#    df_ = df_.applymap(np.floor)
    df_ = np.floor(df_)
    df_ = df_.astype(int)
    df_ = fi*2**(df_/12)
    return df_


def perdida_resolucion(df, a): # la frecuencia incrementa en 1 cada a valores de pixel
    #la frec inicial es fi
    df_ = df/a
    df_ = np.floor(df_)
    df_ = df_.astype(int)
    return df_


def is_within_some_margin(value, lst, margin):
    for item in lst:
        if item == value:
            return item
        if abs(item - value) / value <= margin:
            return item
    return False
    




def play_sound_stereo_ambas(signal_derecha, signal_izquierda, sample_rate, output_filename):
    """
    Reproduce la señal de sonido estéreo utilizando pyaudio.

    Parameters:
    signal (numpy.ndarray): Señal de sonido.
    sample_rate (int): Tasa de muestreo en Hz.
    """
    # Crear una señal estéreo
    stereo_signal = np.zeros((len(signal_derecha), 2))
    stereo_signal[:, 1] = signal_derecha  # Colocar la señal en el canal derecho
    stereo_signal[:, 0] = signal_izquierda  # Colocar la señal en el canal derecho


    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32,
                    channels=2,
                    rate=sample_rate,
                    output=True)
    
    stream.write(stereo_signal.astype(np.float32).tobytes())
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    
    # Guardar la señal en un archivo .wav
    with wave.open(output_filename, 'w') as wf:
        wf.setnchannels(2)
        wf.setsampwidth(4)  # 4 bytes = 32 bits = np.float32
        wf.setframerate(sample_rate)
        wf.writeframes(stereo_signal.astype(np.float32).tobytes())




def save_audio_stereo(signal_derecha, signal_izquierda, sample_rate, output_filename):
    stereo_signal = np.zeros((len(signal_derecha), 2))
    stereo_signal[:, 1] = signal_derecha  # Colocar la señal en el canal derecho
    stereo_signal[:, 0] = signal_izquierda  # Colocar la señal en el canal izquierdo


    # Guardar la señal en un archivo .wav
    with wave.open(output_filename, 'w') as wf:
        wf.setnchannels(2)
        wf.setsampwidth(4)  # 4 bytes = 32 bits = np.float32
        wf.setframerate(sample_rate)
        wf.writeframes(stereo_signal.astype(np.float32).tobytes())


