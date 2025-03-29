## Instructions to run code:

pip install -r requirements.txt

Select the config you want to run by selecting the corresponding config_path in train.py (ll.14-34).

From the root directory (DSF_CODE/), run the following command to train the model:
```
Run python train.py
```

Once the experiments are done, run the following command to generate the plots from the root directory (DSF_CODE/):
``` 
- python plots/plot_ppl_vs_dim.py to generate the plots for the perplexity vs. dimensionality of the hidden layer.
- python plots/plot_ppl_vs_layers.py to generate the plots for the perplexity vs. number of layers.
- python plots/plot_rnn_types_wikitext.py to generate the plots for the perplexity vs. RNN type.
``` 

Note that you can select which plot to generate by changing the run_folders in the corresponding plot file (ll. 43-47).