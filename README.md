# Scalable missingness-avoiding decision trees
This project was completed as part of the WASP course [Scalable Data Science and Distributed Machine Learning](https://lamastex.github.io/scalable-data-science/sds/3/x/) during the fall 2024 semester. The project was collaboratively developed by two contributors: 
 - **Newton Mwai Kinyanjui**, Data Science and AI Division, CSE Department, Chalmers University of Technology
 - **Anton Matsson**, Data Science and AI Division, CSE Department, Chalmers University of Technology.

## Brief project description
   
The project focuses on developing decision trees designed to avoid reliance on missing values during prediction tasks. Such missingness-avoiding (MA) trees are particularly valuable for safety-critical and interpretable machine learning applications like healthcare. Missing values often complicate model deployment, and traditional imputation methods can undermine interpretability.

The core idea of this project is to regularize the node-splitting criterion in decision trees to minimize the presence of missing values along decision paths. To ensure scalability, the project explores implementations of missingness-avoiding (MA) trees using Apache Spark, enabling efficient handling of large datasets. Specifically, it compares three different implementations of an MA tree for classification:
- [`MADTClassifier`](https://github.com/antmats/matrees/blob/main/matrees/estimators.py#L277): A pure Python-based implementation.
- [`PySparkMADTClassifier`](https://github.com/antmats/matrees/blob/main/matrees/estimators.py#L373): A Python-based implementation utilizing RDD operations in PySpark.
- A modified implementation of the Scala [`DecisionTreeClassifier`](https://github.com/antmats/spark/blob/matrees/mllib/src/main/scala/org/apache/spark/ml/classification/DecisionTreeClassifier.scala) class in Spark MLlib, accompanied by a corresponding [Python API](https://github.com/antmats/spark/blob/matrees/python/pyspark/ml/classification.py#L1692).

We compare the different implementations using a synthetic dataset with randomly introduced missing values. Training times are measured for each implementation while varying the number of samples from 100 to 1,000,000, using 20 features. Additionally, we increase the number of features from 10 to 100,000 while keeping the number of samples fixed at 1,000. Finally, we vary the missingness regularization parameter and compare accuracy and missingness reliance across implementations. Each experiment correspond to a certain paramter file (`parameters_01.txt`, `parameters_02.txt` and `parameters_03.txt`). The results of the experiments are presented in the notebook [`results.ipynb`](results.ipynb).

## Running the code

We ran the experiments on the [Tetralith cluster](https://www.nsc.liu.se/systems/tetralith/) provided by the National Supercomputer Centre at Linköping University. To reproduce the results on Tetralith, follow these steps:

1. Clone the project into your home directory:
```bash
cd ~ && git clone https://github.com/antmats/matrees.git
```

2. Build a Singularity container in a storage location with plenty of space:
```bash
cd <path/to/your/project/storage>
mkdir matrees && cd matrees
cp ~/matrees/environment.yml .
cp ~/matrees/container.def .
apptainer build matrees_env.sif container.def
```

3. Update the [Slurm batch script](jobscript.sh) and the parameter files for the experiments with the paths to your project storage. Note that the folders `logs`, `results_01`, `results_02`, and `results_02` are assumed to exist under a folder named `matrees` in the project storage.

4. Launch the experiments by running the following commands:
```bash
cd ~/matrees
sbatch --array=1-15 jobscript.sh parameters_01.txt
sbatch --array=1-15 jobscript.sh parameters_02.txt
sbatch --array=1-75 jobscript.sh parameters_03.txt
```

***

If you want to test the code on your local computer, clone the project into your home directory and install the dependencies in a Conda environment:
```bash
cd ~ && git clone https://github.com/antmats/matrees.git
cd matrees
conda env create -f environment.yml
conda activate matrees_env
```

Next, clone the modified Spark code into your home directory and build Spark using Apache Maven (this may take up to 45 minutes):
```bash
cd ~ && git clone https://github.com/antmats/spark.git
cd spark && git checkout v3.5.3-matrees
./build/mvn clean install -DskipTests -Dscalastyle.skip=true

```

Finally, build a source distribution of PySpark and install it:
```bash
cd python && python setup.py sdist
pip install dist/*.tar.gz
```

Refer to the `main.py` file to launch an experiment:
```bash
python main.py --help
```

## Project presentation

A link to the presentation of the project can be found [here](https://docs.google.com/presentation/d/1jKvugr8AxkjlyV_a55xIcfPLEN_FhEYUKjQ92o0WiYw/edit?usp=sharing).

## Authors' contributions
**Newton:**
- Brainstormed the approach for implementing MA trees using Apache Spark
- Set up the development pipeline with PySpark and Spark MLlib
- Implemented the `PySparkMADTClassifier`
- Wrote shell scripts for running the experiments locally with PySpark and Docker
- Brainstormed changes to the Spark Scala source code, then locally built and debugged the modified Scala source code
- Organized code to align with the experiment pipeline
- Prepared slides for the project pitch and the project presentation
    
**Anton:**
-  Brainstormed the approach for implementing MA trees using Apache Spark
-  Implemented the `MADTClassifier`
-  Implemented the MA trees in Scala by changing the Spark source code
-  Set up comparison experiment for the implementations
-  Set up experiment pipeline with synthetic data on HPC cluster
-  Organized code to align with the experiment pipeline
-  Prepared slides for the project pitch and the project presentation

## Acknowledgements

This project was partially supported by the Wallenberg AI, Autonomous Systems and Software Program funded by Knut and Alice Wallenberg Foundation to fufill the requirements to pass the WASP Graduate School Course Scalable Data Science and Distributed Machine Learning - ScaDaMaLe-WASP-UU-2024 at https://lamastex.github.io/ScaDaMaLe. Computing infrastructure for learning was supported by Databricks Inc.'s Community Edition. The course was Industrially sponsored by Jim Dowling of Logical Clocks AB, Stockholm, Sweden, Reza Zadeh of Matroid Inc., Palo Alto, California, USA, and Andreas Hellander & Salman Toor of Scaleout Systems AB, Uppsala, Sweden.
