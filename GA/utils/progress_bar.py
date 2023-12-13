import sys

def progress_bar(curr_generation, generations, generation_data, exit_condition_score):
  progress_frac = curr_generation/generations
  bar = "#" * int(40 * progress_frac)
  percent = int(100 * progress_frac)

  diff_frac = abs(exit_condition_score)/generation_data.check_diff_last_generations(5)
  arrow = "-" * int(40 * diff_frac) + '>'
  #elapsed = format_time(elapsed)
  msg = "\r[{0:<{1}}] | Generation {2} Completed | [{3:<{1}}] | Exit Condition Progress".format(
      bar, 40, percent+1, arrow#, generation_data.check_diff_last_generations(5) - abs(exit_condition_score)
  )
  sys.stdout.write(msg)
  sys.stdout.flush()