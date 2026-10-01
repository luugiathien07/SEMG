function y = Perfect_Delay(x, d)
  % Circular delay of x by d samples (possibly fractional), via an FFT-domain
  % phase shift -- reconstructed here (not present in the original toolbox
  % directory) from SEMG_simulator_v4.m's own in-line comment that "the
  % Fourier domain is used" to delay each channel's MUAP template by
  % (Channel-1)*DelayFactor samples. Same construction as _shift_fd() in
  % src/mfcv.py (X .* exp(-2j*pi*freqs*tau)), applied here in the sample-index
  % (not physical-frequency) domain since x is a single MUAP template, not a
  % signal with a known Fs.
  N = length(x);
  X = fft(x(:).');
  if mod(N, 2) == 0
    k = [0:N/2-1, -N/2:-1];
  else
    k = [0:(N-1)/2, -(N-1)/2:-1];
  end
  phase = exp(-1i * 2 * pi * k * d / N);
  y = real(ifft(X .* phase));
end
