# Resume–Job Description Matcher Using Transformers 🤖

An intelligent NLP-based application that matches resumes with job descriptions using **Transformer-based semantic embeddings** and **Cosine Similarity**.

## 📌 Project Overview

Traditional resume screening systems mainly depend on keyword matching, which may fail to identify candidates with relevant skills described using different words.

This project uses the **Sentence Transformer model (`all-MiniLM-L6-v2`)** to understand the semantic meaning of resumes and job descriptions, calculate similarity scores, and rank candidates according to their relevance to a particular job role.

## 🎯 Objectives

- Automate the resume screening process.
- Understand the semantic meaning of resume and job description text.
- Calculate similarity scores between resumes and job descriptions.
- Rank candidates based on job relevance.
- Reduce manual effort in initial candidate screening.

## 🛠️ Technologies Used

| Technology            | Purpose                      |
| --------------------- | ---------------------------- |
| Python                | Core programming             |
| Sentence Transformers | Text embeddings              |
| Hugging Face          | Pretrained Transformer model |
| Scikit-learn          | Cosine similarity            |
| Pandas                | Data handling                |
| NumPy                 | Numerical operations         |
| Matplotlib & Seaborn  | Data visualization           |
| Streamlit             | Interactive web application  |

## 🧠 Transformer Model

**Model:** `all-MiniLM-L6-v2`

The pretrained Sentence Transformer converts resumes and job descriptions into numerical vectors called embeddings.

These embeddings capture semantic information, allowing the system to identify meaningful similarities even when different words are used.

## ⚙️ Project Workflow

1. **Data Collection:** Collect resume descriptions and job descriptions.
2. **Text Preprocessing:** Prepare and clean textual information.
3. **Embedding Generation:** Convert text into numerical vectors using Sentence Transformers.
4. **Similarity Calculation:** Apply cosine similarity to compare resume and job embeddings.
5. **Candidate Ranking:** Rank candidates based on their similarity scores.
6. **Visualization:** Display matching results using charts and graphs.

## ✨ Key Features

- Transformer-based semantic matching.
- Multiple resume and job description comparison.
- Similarity score calculation.
- Candidate ranking.
- Matching score visualization.
- Support for expanding the dataset.

## 📊 Expected Output

The system generates a similarity matrix representing the matching scores between resumes and job descriptions.

It can also identify the most relevant resumes for a selected job description.

## 📂 Project Structure

```text
Resume-Job-Description-Matcher-Transformer/
│
├── data/
├── notebooks/
├── src/
├── app/
├── results/
├── screenshots/
├── README.md
├── requirements.txt
└── .gitignore
```

## 🚀 Installation and Execution

### 1. Clone the repository

```bash
git clone https://github.com/pooja21-chaturvedi/Resume-Job-Description-Matcher-Transformer.git
```

### 2. Navigate to the project folder

```bash
cd Resume-Job-Description-Matcher-Transformer
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Jupyter Notebook

```bash
jupyter notebook
```

Open the notebook inside the `notebooks` folder and execute the cells.

## 🔮 Future Enhancements

- Upload resumes in PDF format.
- Automatic skill extraction.
- Experience and education-based matching.
- Interactive Streamlit dashboard.
- Advanced candidate ranking and evaluation.
- Integration with real-world recruitment datasets.

## ⚠️ Disclaimer

Similarity scores represent semantic relevance between resumes and job descriptions. They should not be considered a hiring probability or the sole basis for recruitment decisions.

## 👩‍💻 Author

**Pooja Chaturvedi**

**Isha Bhatnagar**

**Nikhil Yadav**

B.Tech – Computer Science and Information Technology

GitHub: [pooja21-chaturvedi](https://github.com/pooja21-chaturvedi)

---

⭐ If you find this project useful, consider giving the repository a star!

