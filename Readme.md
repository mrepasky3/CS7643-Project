# Project for Deep Learning Class

Three datasets for set-based classification problems:  
1. Toy Data  
    * Sum over digits  
    * Max over digits  
    * Lowest-valued mode of set of digits  
    * Product of two largest digits in a set  
    * Max - Min of digits  
2. MNIST Data  
    * Sum over digits  
    * Max over digits  
    * Lowest-valued mode of set of digits  
    * Product of two largest digits in a set  
    * Max - Min of digits  
3. Omniglot Data  
    * Number of unique symbols in the set  

The plots of the training loss, generalization to various fixed set sizes, and batch processing times for various fixed step sizes are in the following table:
| Task | Training     | Generalization    | Batch Time |
|:----------------------------:|:----------------------------:|:----------------------------:|:----------------------------:|
| Toy Sum | <img src="https://github.com/mrepasky3/CS7643-Project/blob/main/simple_results/Toy_sum/training_curves.png" width="270" height="210" /> | <img src="https://github.com/mrepasky3/CS7643-Project/blob/main/simple_results/Toy_sum/fixed_size_curves.png" width="270" height="210" /> | <img src="https://github.com/mrepasky3/CS7643-Project/blob/main/simple_results/Toy_sum/batch_time_curves.png" width="270" height="210" /> |
| Toy Max | <img src="https://github.com/mrepasky3/CS7643-Project/blob/main/simple_results/Toy_max/training_curves.png" width="270" height="210" /> | <img src="https://github.com/mrepasky3/CS7643-Project/blob/main/simple_results/Toy_max/fixed_size_curves.png" width="270" height="210" /> | <img src="https://github.com/mrepasky3/CS7643-Project/blob/main/simple_results/Toy_max/batch_time_curves.png" width="270" height="210" /> |
| Toy Mode | <img src="https://github.com/mrepasky3/CS7643-Project/blob/main/simple_results/Toy_mode/training_curves.png" width="270" height="210" /> | <img src="https://github.com/mrepasky3/CS7643-Project/blob/main/simple_results/Toy_mode/fixed_size_curves.png" width="270" height="210" /> | <img src="https://github.com/mrepasky3/CS7643-Project/blob/main/simple_results/Toy_mode/batch_time_curves.png" width="270" height="210" /> |
| Toy Product | <img src="https://github.com/mrepasky3/CS7643-Project/blob/main/simple_results/Toy_product/training_curves.png" width="270" height="210" /> | <img src="https://github.com/mrepasky3/CS7643-Project/blob/main/simple_results/Toy_product/fixed_size_curves.png" width="270" height="210" /> | <img src="https://github.com/mrepasky3/CS7643-Project/blob/main/simple_results/Toy_product/batch_time_curves.png" width="270" height="210" /> |
| Toy Range | <img src="https://github.com/mrepasky3/CS7643-Project/blob/main/simple_results/Toy_range/training_curves.png" width="270" height="210" /> | <img src="https://github.com/mrepasky3/CS7643-Project/blob/main/simple_results/Toy_range/fixed_size_curves.png" width="270" height="210" /> | <img src="https://github.com/mrepasky3/CS7643-Project/blob/main/simple_results/Toy_range/batch_time_curves.png" width="270" height="210" /> |
| MNIST Sum | <img src="https://github.com/mrepasky3/CS7643-Project/blob/main/simple_results/MNIST_sum/training_curves.png" width="270" height="210" /> | <img src="https://github.com/mrepasky3/CS7643-Project/blob/main/simple_results/MNIST_sum/fixed_size_curves.png" width="270" height="210" /> | <img src="https://github.com/mrepasky3/CS7643-Project/blob/main/simple_results/MNIST_sum/batch_time_curves.png" width="270" height="210" /> |
| MNIST Max | <img src="https://github.com/mrepasky3/CS7643-Project/blob/main/simple_results/MNIST_max/training_curves.png" width="270" height="210" /> | <img src="https://github.com/mrepasky3/CS7643-Project/blob/main/simple_results/MNIST_max/fixed_size_curves.png" width="270" height="210" /> | <img src="https://github.com/mrepasky3/CS7643-Project/blob/main/simple_results/MNIST_max/batch_time_curves.png" width="270" height="210" /> |
| MNIST Mode | <img src="https://github.com/mrepasky3/CS7643-Project/blob/main/simple_results/MNIST_mode/training_curves.png" width="270" height="210" /> | <img src="https://github.com/mrepasky3/CS7643-Project/blob/main/simple_results/MNIST_mode/fixed_size_curves.png" width="270" height="210" /> | <img src="https://github.com/mrepasky3/CS7643-Project/blob/main/simple_results/MNIST_mode/batch_time_curves.png" width="270" height="210" /> |
| MNIST Product | <img src="https://github.com/mrepasky3/CS7643-Project/blob/main/simple_results/MNIST_product/training_curves.png" width="270" height="210" /> | <img src="https://github.com/mrepasky3/CS7643-Project/blob/main/simple_results/MNIST_product/fixed_size_curves.png" width="270" height="210" /> | <img src="https://github.com/mrepasky3/CS7643-Project/blob/main/simple_results/MNIST_product/batch_time_curves.png" width="270" height="210" /> |
| MNIST Range | <img src="https://github.com/mrepasky3/CS7643-Project/blob/main/simple_results/MNIST_range/training_curves.png" width="270" height="210" /> | <img src="https://github.com/mrepasky3/CS7643-Project/blob/main/simple_results/MNIST_range/fixed_size_curves.png" width="270" height="210" /> | <img src="https://github.com/mrepasky3/CS7643-Project/blob/main/simple_results/MNIST_range/batch_time_curves.png" width="270" height="210" /> |
| Omni Unique | <img src="https://github.com/mrepasky3/CS7643-Project/blob/main/simple_results/OMNI_unique/training_curves.png" width="270" height="210" /> | <img src="https://github.com/mrepasky3/CS7643-Project/blob/main/simple_results/OMNI_unique/fixed_size_curves.png" width="270" height="210" /> | <img src="https://github.com/mrepasky3/CS7643-Project/blob/main/simple_results/OMNI_unique/batch_time_curves.png" width="270" height="210" /> |

# References
- Sepp Hochreiter and Jurgen Schmidhuber. Long short-term memory. *Neural computation*, 9(8):1735–1780, 1997.
- Brenden M Lake, Ruslan Salakhutdinov, and Joshua B Tenenbaum. Human-level concept learning through probabilistic program induction. *Science*, 350(6266):1332–1338, 2015.
- Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In *Advances in neural information processing systems*, 2017.
- Manzil Zaheer, Satwik Kottur, Siamak Ravanbakhsh, Barnabas Poczos, Ruslan Salakhutdinov, and Alexander Smola. Deep sets. In *Advances in neural information processing systems*, 2017.
