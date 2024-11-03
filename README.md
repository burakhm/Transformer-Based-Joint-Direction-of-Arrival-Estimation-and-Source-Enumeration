<h1 align='center'>
    Transformer Based Joint Direction of Arrival Estimation and Source Enumeration Robust to Sensor Failures in Sparse Linear Arrays
</h1>

<div>
    <p align='center'>
        Burak Hayati Muslu<sup>1,2</sup>&emsp;
        Tolga Çiloğlu<sup>1</sup>
    </p>
</div>

<div>
    <p align='center'>
        <sup>1</sup>Dept. of Electrical and Electronics Eng., METU, Ankara, Türkiye<br>
        <sup>2</sup>Surveillance Radar Systems Engineering Department, ASELSAN, Ankara, Türkiye
    </p>
</div>



## Installation
Create conda environment:

```bash
  conda create -n doaformer python=3.10
  conda activate doaformer
```

Install packages using 'pip':

```bash
  pip install -r requirements.txt
```

## Data Generation
Generate training and test data for each experiment:

```bash
  cd scripts
  generate_data.bat
  cd ..
```

## Training and Evaluation
Run the notebook files which are located in 'script' folder for each experiment and scenario. 
Generate evaluation graphs using 'compare' notebook. 
Information about experiments and scenarios are as follows:

- **Experiment 1**: Different sparse array 
    - **Scenario 1**: Minimum redundant array
    - **Scenario 2**: Nested array
    - **Scenario 3**: Coprime array
	
- **Experiment 2**: Sensor malfunctions
    - **Scenario 1**: Intact array
    - **Scenario 2**: Faulty array
	
- **Experiment 3:** Unknown number of sources