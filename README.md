## Project structure

.
├── CelebA_HQ_facial_identity_dataset
│   ├── CelebA-HQ-to-CelebA-mapping.txt
│   ├── list_identity_celeba.txt
│   ├── test
│   └── train
├── Data_generation.ipynb
├── environment.yml
├── Facial_Detection.ipynb
├── Final_presentation_TensorOverflow.pdf
├── GUI.py
├── history
│   ├── train_history_307.csv
│   ├── train_history_307.svg
│   ├── train_history_310.csv
│   ├── train_history_310_plus_max_pro_ultra.csv
│   ├── train_history_310_plus_max_pro_ultra.svg
│   ├── train_history_310.svg
│   └── train_history_3.csv
├── inference.ipynb
├── live_demo.py
├── models
│   ├── class_names_310.json
│   ├── model_307.pth
│   ├── model_310_plus_max_pro_ultra.pt
│   ├── model_310.pth
│   └── model_3.pth
├── Preprocessing.ipynb
├── Project proposal.pdf
├── README.md
├── Report
│   ├── Appendix.tex
│   ├── Figures
│   ├── references.bib
│   ├── Report.pdf
│   ├── Report.synctex.gz
│   ├── Report.tex
│   ├── report_tex_file.zip
│   ├── Section1.tex
│   ├── Section2.tex
│   ├── Section3.tex
│   ├── Section4.tex
│   ├── Section5.tex
│   └── uOttawa.png
├── Report.pdf
├── requirement.txt
├── simple_model.ipynb
└── utils.py

Due to the file size limit of GitHub, the dataset is not included in this repository. Please download the dataset from the following link and put it in the same directory as the code: 
https://drive.google.com/drive/folders/1fjAtXjUwjHe6Y2LkrpFL7Uzp8FyYHmyC?usp=sharing

## Instruction of re-construct environment

Check your current folder:

```shell
pwd
```

It should be:

```shell
../FaceEncryption
```

Re-construct environment by using virtualenv

```shell
python3 -m pip install --user --upgrade pip

python3 -m pip install --user virtualenv

python3 -m venv TensorOverflow

source TensorOverflow/bin/activate

python3 -m pip install -r requirements.txt
```

**You can now execute the live_demo.py to test this project**

