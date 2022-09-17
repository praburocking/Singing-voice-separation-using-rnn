#!/usr/bin/env python
# -*- coding: utf-8 -*-

import librosa
import numpy as np

__docformat__ = 'reStructuredText'
__all__ = ['to_audio']


def to_audio(mix_waveform: np.ndarray, 
             predicted_vectors: np.ndarray) \
    -> np.ndarray:
	"""
	:param mix_waveform: The waveform of the monaural mixture. Expected shape (n_samples,)
    :type mix_waveform: numpy.ndarray
	:param predicted_vectors: A numpy array of shape: (chunks, frequency_bins, time_frames)
    :type predicted_vectors: numpy.ndarray
	:return: predicted_waveform: The waveform of the predicted signal: (~n_samples,)
    :rtype: numpy.ndarray
	"""
	# Pre-defined (I)STFT parameters
	win_size = 2048
	hop_size = win_size // 2
	win_type = 'hamm'

	# STFT analysis of waveform
	c_x = librosa.stft(mix_waveform, n_fft=win_size, win_length=win_size, hop_length=hop_size, window=win_type)
	# Phase computation
	phs_x = np.angle(c_x)
	# Get the number of time-frames
	tf = phs_x.shape[1]

	# Number of chunks/sequences
	n_chunks, fb, seq_len = predicted_vectors.shape
	p_end = seq_len*n_chunks
	# Reshaping
	rs_vectors = np.reshape(predicted_vectors, (fb, p_end))
	# Reconstruction
	if p_end > tf:
		# Appending zeros to phase
		c_vectors = np.hstack((phs_x, np.zeros_like(phs_x[:, :p_end-seq_len])))
	else:
		c_vectors = rs_vectors * np.exp(1j * phs_x[:, :p_end])
	# ISTFT
	predicted_waveform = librosa.istft(c_vectors, win_length=win_size, hop_length=hop_size, window=win_type)


	return predicted_waveform


def main():
	# Make a test
	seq_length = 60
	sig_len = 1898192
	mix_sig = np.random.normal(0, 0.8, (sig_len,))
	c_x = librosa.stft(mix_sig, n_fft=2048, win_length=2048, hop_length=1024, window='hamm')
	chop_factor = c_x.shape[1] % seq_length
	new_time_frames = c_x.shape[1] - chop_factor

	# A sketch for the chunked sequences, that contain the magnitude estimates of the signal
	r_vectors = np.reshape(np.abs(c_x[:, :-chop_factor]), (new_time_frames//seq_length, c_x.shape[0], seq_length))
	print(r_vectors.shape)
	print(mix_sig.shape)
	rec_wav = to_audio(mix_sig, r_vectors)
	print("MSE: %f" % np.mean((rec_wav - mix_sig[:len(rec_wav)])**2.))

	return None


if __name__ == '__main__':
	main()

# EOF
