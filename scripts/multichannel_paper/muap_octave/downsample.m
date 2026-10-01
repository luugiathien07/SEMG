function y = downsample(x, n)
  % Minimal stand-in for the Signal Processing Toolbox's downsample():
  % keep every n-th sample, starting from the first. No Octave 'signal'
  % package is installed in this environment (pkg list is empty), so
  % SEMG_simulator_v4.m's call to downsample() has nothing to resolve to
  % without this.
  x = x(:).';
  y = x(1:n:end);
end
