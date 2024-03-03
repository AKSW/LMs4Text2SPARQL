# LMs 4 Text2SPARQL

This repository contains code to finetune a varietey of LMs with under one billion parameters on different datasets.

## How to use

There are three datasets available right now:
- The organizational demo graph
- Sample data from Coypu
- LC_QUAD

To run the training for 20 epochs on the coypu dataset for example, execute the following:

  ```bash
python train.py --num-epochs 20 --dataset coypu
  ```

The code will stop every five iterations to check how the model performs. This will create json-Files under `./results` that contain the generated SPARQL query as well as the gold standard. To actually evaluate the results, run the following:

  ```bash
python eval.py --num-epochs 20 --dataset coypu
  ```

The dataset parameter is important to let the eval script know which dataset to run the queries on. The number of epochs are specified here as well to tell the script how many json files to expect per model. It will print some stuff to the screen
which was only used during development, you can ignore that. After the script is done, you will find a file called `total_{dataset}.json` in the results folder, for example `./results/total_coypu.json` which contains for each model the number of
correctly translated questions.

### Docker
With `make` you get an overview of all the avialable tasks. 
First you would need to build the docker image localy. 

```bash
docker build -t akws/lms4text2sparql .
```

Then you can run the benchmark for a given dataset with:

```bash
make run-orga
```