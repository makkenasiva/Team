from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load your dataset from the CSV file
csv_file_path = r"data/Book1.csv"
data = pd.read_csv(csv_file_path)

# Initialize the TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()

def find_similar_responses_and_grade(new_response, question_id, data, tfidf_vectorizer, rubrics, correct_ans, max_response):
    filtered_data = data[data['QuestionID'] == question_id].copy()

    # If there are no existing responses, compare with the correct answer
    if filtered_data.empty:
        tfidf_matrix = tfidf_vectorizer.fit_transform([correct_ans])
    else:
        tfidf_matrix = tfidf_vectorizer.fit_transform(filtered_data['StudentResponse'])

    new_response_vector = tfidf_vectorizer.transform([new_response])
    cosine_similarities = cosine_similarity(new_response_vector, tfidf_matrix).flatten()

    if not filtered_data.empty:
        filtered_data['CosineSimilarity'] = cosine_similarities
        similar_responses = filtered_data.sort_values(by='CosineSimilarity', ascending=False)
    else:
        similar_responses = pd.DataFrame({'CosineSimilarity': cosine_similarities})

    max_similarity = round(similar_responses.iloc[0]['CosineSimilarity'], 2)

    if new_response == correct_ans:
        assigned_grade = max_response
    elif max_similarity == 1:
        assigned_grade = similar_responses.iloc[0]['StudentScore'] / similar_responses.iloc[0]['MaxPossibleScore'] * max_response
    elif max_similarity >= 0.75:
        assigned_grade = (similar_responses.iloc[0]['StudentScore'] / similar_responses.iloc[0]['MaxPossibleScore']) * max_response * max_similarity
    else:
        tfidf_matrix_correct = tfidf_vectorizer.fit_transform([correct_ans])
        new_response_vector_correct = tfidf_vectorizer.transform([new_response])
        cosine_similarity_correct = cosine_similarity(new_response_vector_correct, tfidf_matrix_correct).flatten()[0]
        assigned_grade = round(cosine_similarity_correct * max_response)

    save_new_data(question_id, new_response, max_response, assigned_grade, rubrics)
    
    return f"The new response has a similarity of {max_similarity}. Assigned grade: {round(assigned_grade)}"

def save_new_data(question_id, new_response, max_response, assigned_grade, rubrics):
    new_data = pd.DataFrame({
        "Question": ["FILLER"],
        "QuestionID": [question_id],
        "Identifier": ["FILLER"],
        "StudentResponse": [new_response],
        "CorrectAnswer": ["FILLER"],
        "MaxPossibleScore": [max_response],
        "StudentScore": [round(assigned_grade)],
        "TenantName": ["FILLER"],
        "Provider": ["FILLER"],
        "Rubric": [rubrics],
        "Grade": ["FILLER"],
        "Subject": ["FILLER"]
    })
    
    new_data.to_csv("data/Book1.csv", mode='a', header=False, index=False)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    json_data = request.get_json()
    student_response = json_data['studentResponse']
    question_id = int(json_data['questionId'])
    rubrics = json_data['rubrics']
    max_response = int(json_data['maxResponse'])
    correct_ans = json_data['correctAnswer']

    result = find_similar_responses_and_grade(student_response, question_id, data, tfidf_vectorizer, rubrics, correct_ans, max_response)

    score = None
    if 'Assigned grade' in result:
        score = result.split(': ')[-1]
    return jsonify({'message': result, 'score': score})

@app.errorhandler(500)
def internal_server_error(e):
    return jsonify(error=str(e)), 500

if __name__ == '__main__':
    app.run(debug=True)
