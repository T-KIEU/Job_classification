# Job_classification
Career level classification for 8k jobs.


## Dataset
Here is an example of the dataset:

| title | location | description | function | industry | career_level |
| --- | --- | --- | --- | --- | --- |
| Technical Professional Lead - Process | Houston, TX | "Responsible for the study, design, and specification and services related to overpressure protection of chemical facilities, including flare headers and relief devices. In the context of overpressure protection, candidate should also be capable of leading other engineers to develop Â heat and material balance, the development of process flow diagrams, piping and instrument diagrams, system design parameters, and design or rating of equipment and apparatus such as heat exchangers, vessels, reactors, etc. May also be responsible for overseeing development of process simulations and advanced controls/process optimization. Provides technical guidance to less experienced engineers. Plans, schedules, and conducts working requiring independent judgment and adaptation of engineering techniques. The successful candidate for this position enjoys working with people and has the majority of their experience with a major EPC company and not at a production facility. Qualifications Requirements Bachelors or Masters degree in Chemical Engineering, with a minimum of ten (10) years experience, and a typical experience of fifteen (15) years plus. Related experience, including P&amp;ID development, relief valve sizing, and hydraulic calculations. Knowledge of SmartPlant P&amp;ID, Pro-II, and Unisim/Aspen software suites is desired. Experience leading small to large design groups is required. | production_manufacturing | Machinery and Industrial Facilities Engineering | senior_specialist_or_project_manager |


## Pre-processing
A pipeline of preprocessing: <br />
* Remove missing values <br />
* Transform data: TfidfVectorizer, OneHotEncoder <br />
<br />


## Training
Random Forest Classifier model <br />
Parameters: estimators, criterion, selection_percentile, min_df, max_df, ngram_range <br />
<br />


## Performances
![image](https://github.com/T-KIEU/Job_classification/assets/100022674/88c8b971-399e-4ddd-b390-08120680d25a)


