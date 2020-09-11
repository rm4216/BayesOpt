#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
'''
Utility methods for interfacing with the user
and/or for development purposes.

Authored: 2017-03-12
Modified: 2017-10-24
'''

# -------- Dependencies
import os, sys
import time, json
import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline

from pdb import set_trace as bp


__all__ =\
[
  'session_profiler',
  'uninitialized_variables',
  'progress_bar',
  'make_eq_test',
  'bp',
]

# ==============================================
#                                       user_ops
# ==============================================
class session_profiler(object):
  def __init__(self, session=None, options=None, run_metadata=None,
    profile=None, save_dir=None):
    if (options is None):
      options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    self.options = options

    if (run_metadata is None):
      run_metadata = tf.RunMetadata()
    self.run_metadata = run_metadata

    self.profile = profile
    self.save_dir = save_dir
    self.session = session

  def run(self, *args, session=None, options=None, run_metadata=None, 
    update=True, **kwargs):
    if (session is None): session = self.session
    if (options is None): options = self.options
    if (run_metadata is None): run_metadata = self.run_metadata

    fetches = session.run(*args, options=options, 
                  run_metadata=run_metadata, **kwargs)

    if update: self.update(run_metadata)
    return fetches

  def update(self, run_metadata=None):
    if (run_metadata is None):
      run_metadata = self.run_metadata

    # Get run statistics as timeline and convert to dict
    fetched_timeline = timeline.Timeline(run_metadata.step_stats)
    chrome_trace = fetched_timeline.generate_chrome_trace_format()    
    trace_dict = json.loads(chrome_trace)

    # For first run store full trace
    if (self.profile is None):
      self.profile = trace_dict
    else:
      # Event runtimes are prefixed 'ts'
      runtimes = filter(lambda event: 'ts' in event, trace_dict['traceEvents'])
      self.profile['traceEvents'].extend(runtimes)

  def save(self, filename):
    if (self.save_dir is not None):
      filename = '/'.join([self.save_dir, filename])

    # Ensure existance of parent directories
    address = filename.split('/')
    filepath = ''
    for k in range(len(address)-1):
      filepath += address[k]
      if filepath and not os.path.exists(filepath):
        os.makedirs(filepath)
      filepath += '/'

    with open(filename, 'w') as file:
        json.dump(self.profile, file)


def uninitialized_variables(sess, *args, filter_op=None, **kwargs):
  '''
  Return a list of uninitialized variables for a given session
  '''
  global_vars = tf.global_variables()
  if (filter_op is not None):
    global_vars = filter(filter_op, global_vars)

  uninitialized_vars = []
  for var in global_vars:
    if not sess.run(tf.is_variable_initialized(var), *args, **kwargs):
     uninitialized_vars.append(var)
  return uninitialized_vars


def progress_bar(progress, duration=None, title='Progress', 
  char='#', precision=1, bar_length=50):
  assert isinstance(progress, float), "Progress value must be <float>-typed"
  num_chars = int(round(bar_length * progress))
  bar = num_chars * char + ' '*(bar_length - num_chars*len(char))
  msg = '\r{:s}: [{:s}] {:.{p}f}%'.format(title, bar, 100*progress, p=precision)
  if (duration is not None):
    eta = (1 - progress)/progress * duration
    msg += ', ETA: {:.2e}s'.format(eta)
  if progress >= 1.0: msg += '\r\n'
  sys.stdout.write(msg)
  sys.stdout.flush()


def make_eq_test(A, B, tol=0.0, name=None):
  with tf.name_scope('make_eq_test') as scope:
    tol = tf.constant(tol, A.dtype)
    return tf.assert_less(tf.reduce_max(tf.abs(A - B)), tol, name=name)


# ==============================================
#                            Developers' Section
# ==============================================
# ------------ Scrap work goes here ------------
'''
'''