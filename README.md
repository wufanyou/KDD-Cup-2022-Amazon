# KDD-Cup-2022-Amazon
This repository is the team ETS-Lab's solution for Amazon KDD Cup 2022. You can find our code submission [here](https://gitlab.aicrowd.com/wufanyou/kdd_task_2) or check the solution paper [here](https://arxiv.org/abs/2208.00108).


#### General solution

* We trained __6 cross encoder models__ for each language which differs in the pertained models, training method (e.g., knowledge distillation), and data splitting. In total, six identical models (2 folds x 3 models) are used to produce the initial prediction (4 class probability) of the query-product pair. Using those models only, the public set score for task 2 is around __0.816__.

* For Task 1, we used the output 4 class probability with some simple features to train a lightgbm model, calculate the __expected gain__ ($P_e\*1 + P_s\*0.1 + P_c\*0.01$), and sort the query-product list by this gain.

* For task 2 and Task 3, we used lightgbm to fuse those prediction with __some important features__. Most important features are designed based on the potential data leakage from task 1 and the behavior of the query-product group:
    * The stats (min, medium and max) of the cross encoder output probability grouped by query_id (0.007+ in Task 2 Public Leaderboard)
    * The percentage of product_id in Task 1 product list grouped by query_id (0.006+ in Task 2 Public Leaderboard) 


#### Small modification towards Cross Encoder architecture

* As the product context has multiple fields (title, brand, and so on), we use neither the cls token nor mean (max) pooling to get the latent vector of the query-product pair. Instead, we concatenate the hidden states of a predefined token (query, title, brand color, etc.). The format is: 
    ```
    [CLS] [QUERY] <query_content> [SEP] [TITLE] <title_content> [SEP] [Brand] <brand_content> [SEP] ...
    ```
    where `[TEXT]` is the special token and `<text_content>` is the text contents.

#### Code submission speed up
1. Pre-process product token and saved as a HDF5 file.
2. Transfer all model to ONNX with FP16 precision.
3. Pre-sorted the product id to reduce the side impact of batch zero padding.
4. Use relative small mini-batch size when inference.
#### How to run code

* You need to write down some config.yaml file

* For training:
    ```python
    python train.py -c config/us-bigbird-kd-0.yaml
    ```
* For inference:

    ```python
    python inference.py -c config/us-bigbird-kd-0.yaml -w last -ds test
    ```
