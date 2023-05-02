# Medical Imaging - Group 5

# Project Status

**Code**

-   [x] Compactness
-   [x] Asymmetry
-   [ ] Fix the center function, so it can support multiple images and lesion types
-   [ ] Make color extraction correctly account for blue
-   [ ] SLIC thing and the formula on messenger
-   [ ] Make code that writes all data onto the metadata.csv
-   [ ] Make sure the code can run her evaluation script
-   [ ] Clean up github repo and the code
-   [ ] Make model

**Report**

-   [x] Introduction
-   [ ] ?
-   [ ] ?
-   [ ] ?
-   [ ] Spell check the report

# Features

-   Color

## Dataset

-   patient_id
    -   Unique identifier for each patient
-   lesion_id
    -   Unique identifier for each lesion
-   smoke
    -   If the patient smokes
-   drink
    -   If the patient drinks
-   background_father
    -   The ethnicity of the patient's father
-   background_mother
    -   The ethnicity of the patient's mother
-   age
    -   The age of the patient
-   pesticide
    -   If the cancer is caused by a pesticide
-   gender
    -   The gender of the paient
-   skin_cancer_history
    -   If the patient has a history of skin cancer
-   cancer_history
    -   If the patient has a history of cancer
-   has_piped_water
    -   Whether or not the patient has access to piped water
-   has_sewage_system
    -   If the patient has a sewage system
-   fitspatrick
    -   Skin color of the paitient
-   region
    -   Which region the patient is from
-   diameter_1
-   diameter_2
-   diagnostic
    -   Basal Cell Carcinoma (BCC)
    -   Squamous Cell Carcinoma (SCC)
    -   Actinic Keratosis (ACK)
    -   Seborrheic Keratosis (SEK)
    -   Bowen’s disease (BOD)
    -   Melanoma (MEL)
    -   Nevus (NEV)
    -   As the Bowen’s disease is considered SCC in situ, we clustered them together, which results in six skin lesions in the dataset, three skin cancers (BCC, MEL, and SCC) and three skin disease (ACK, NEV, and SEK)
-   itch
    -   If the spot itches
-   grew
    -   If the spot grew
-   hurt
    -   If the spot hurts
-   changed
-   bleed

    -   If the spot bleeds

-   elevation
    -   If it extrudes from the skin
-   img_id
    -   The id of the image
-   bio
    psed
