# josie_trrm_analysis
Analysis code for JOSIE TRRM 2023 paper
## to run over 0910 data
1) make_df/josie0910_make_data.py
run 2 times with setting year == '2009' and year == '2010'
2) run merge_df to combine 2009 and 2010
3) run add_preparation_data to add preparation data
4) make_beta_convoluted_data with bool_0910_decay = True and year = '0910' -> this applies the trrm method on the simulations
that are not in preparation files. The decay_time for these files are taken as the median of available decay times
5) make_beta_convoluted_predata -> this code runs trrm method on 0910 data that has preparation information
6) make_final_0910_data -> this code merges output of 4 with output of 5, the output of this file is the final 0910
to be used for analysis
## to run over 2017 data
1) make_df/josie17_make_data
2) make_beta_convoluted_data.py



