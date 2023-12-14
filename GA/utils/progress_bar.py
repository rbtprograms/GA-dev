import sys

def progress_bar(curr_generation, generations, generation_data):
  progress_frac = curr_generation/generations
  bar = "-" * int(40 * progress_frac) + '>'
  percent = int(100 * progress_frac)


  msg = "\r[{0:<{1}}] | Generation {2} Completed".format(
      bar, 40, percent+1
  )
  sys.stdout.write(msg)
  sys.stdout.flush()