# Labeling Guidelines for Mental Health Conversations

This document provides guidelines for annotating mental health conversations with the following labels: **Anxiety**, **Depression**, and **Stress**, based on the ICD-11 framework. Some posts may receive multiple labels with different confidence scores, while others may fall into the **Other** category if they do not clearly match any of the three classes.

## Important Note for Annotators
Some posts in this dataset may contain content that could be distressing or emotionally triggering, particularly regarding topics like depression or suicidal thoughts. If you feel uncomfortable or sensitive to these topics, participation in the annotation of such posts is optional. Please prioritize your mental well-being, and skip any posts that you find too distressing to label. 
## Label Categories and Guidelines

### 1. Anxiety
- **ICD-11 Code**: 6B00 (Generalized Anxiety Disorder)
- **Key Symptoms**:
  - Excessive worry and anxiety about multiple events or activities.
  - Difficulty controlling worry.
  - Restlessness, fatigue, difficulty concentrating, irritability, muscle tension, or sleep disturbance.
- **Diagnostic Criteria**: The symptoms should persist for at least several months and cause significant distress or impairment in daily functioning.
- **Guideline for Labeling**: 
  - Label conversations as **Anxiety** if the individual expresses ongoing worry, fear, or nervousness about various life events or activities.
  - Conversations indicating physical symptoms such as restlessness or difficulty sleeping should also be considered.

### 2. Depression
- **ICD-11 Code**: 6A70 (Depressive Episode)
- **Key Symptoms**:
  - Persistent sadness, loss of interest or pleasure in activities.
  - Fatigue or loss of energy.
  - Feelings of worthlessness or excessive guilt.
  - Thoughts of death or suicide.
  - Difficulty concentrating, indecisiveness.
- **Diagnostic Criteria**: Symptoms should last for at least two weeks, and cause significant impairment in personal, social, or occupational functioning.
- **Guideline for Labeling**:
  - Label conversations as **Depression** if the speaker mentions feeling persistently sad, losing interest in activities they once enjoyed, or experiencing feelings of guilt or worthlessness.
  - Mention of suicidal thoughts or plans should automatically trigger a label of **Depression**.

### 3. Stress
- **ICD-11 Code**: MB23 (Response to Severe Stress)
- **Key Symptoms**:
  - Emotional or psychological stress in response to external factors (e.g., work pressure, personal crises).
  - Symptoms of irritability, anger, frustration, or feelings of being overwhelmed.
  - Physical symptoms such as headaches or fatigue.
- **Diagnostic Criteria**: Stress symptoms must be linked to an identifiable event or stressor, and may vary in intensity based on the severity of the trigger.
- **Guideline for Labeling**:
  - Label conversations as **Stress** if the individual discusses feeling overwhelmed, frustrated, or irritable due to external circumstances (e.g., work, family problems).
  - Stress-related physical symptoms like headaches or fatigue can also indicate this label.

## 4. Multi-labeling with Confidence Scores
Some posts may fit into multiple categories. In such cases, annotators should assign a percentage score for each label based on how strongly the post aligns with each category. For example:
- A post may be labeled as **60% Anxiety** and **40% Stress** if the individual expresses both excessive worry and overwhelming feelings related to external stressors.

### 5. Other
- Label posts as **Other** if the content does not clearly fit into **Anxiety**, **Depression**, or **Stress** categories. This may include conversations about other mental health issues (e.g., PTSD, bipolar disorder), or non-mental health-related topics.
  
## Example Annotations
- **Anxiety Example**: "I can’t stop worrying about my upcoming exams, and it’s affecting my sleep." → Label: **Anxiety**
- **Depression Example**: "I don’t feel like doing anything anymore. Even things I used to enjoy are meaningless now." → Label: **Depression**
- **Stress Example**: "I’ve been so overwhelmed at work lately, and it’s giving me headaches." → Label: **Stress**
- **Multi-label Example**: "I’m feeling extremely anxious about my family’s financial situation, and it’s causing me a lot of stress." → Label: **70% Anxiety**, **30% Stress**
- **Other Example**: "I’ve been having flashbacks from a traumatic event." → Label: **Other**

## Contact Information
For any questions or clarification, please contact me at saadaoui@umich.edu.

---

