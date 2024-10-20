from flask import Flask,render_template, request, session, redirect , jsonify, Response
from werkzeug.utils import secure_filename
from flask_mysqldb import MySQL
from flask import Flask, render_template, request, session, redirect
from flask_socketio import SocketIO, emit
from flask.helpers import url_for
import mysql.connector
import csv
import os
import io
import string
import random
import datetime
import requests
import bcrypt
import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# Mock database to store previous messages
messages = []
app = Flask(__name__)
app.secret_key=os.urandom(24)

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = '1234'
app.config['MYSQL_DB'] = 'dupe'
mysql = MySQL(app)

app.config['UPLOAD_FOLDER'] = './static/uploads/'
socketio = SocketIO(app)

model = None
scaler = None

def load_models():
    """Load models with proper error handling and version compatibility"""
    global model, scaler
    
    try:
        # Try to load the model with explicit scikit-learn version handling
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = joblib.load("voting_classifier_model.pkl")
            scaler = joblib.load("scaler.pkl")
            
        print("Models loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading models: {e}")
        return False

def retrain_model():
    """Retrain the model with current scikit-learn version"""
    global model, scaler
    try:
        # Your model training code here
        # Make sure to use the same parameters as the original model
        # Save the newly trained model
        joblib.dump(model, "voting_classifier_model.pkl")
        joblib.dump(scaler, "scaler.pkl")
        return True
    except Exception as e:
        print(f"Error retraining model: {e}")
        return False

def process_data(df):
    """Process input data for prediction"""
    try:
        # Create features similar to training data
        df['hour'] = pd.to_numeric(df['hour'], errors='coerce')
        df['minutes_used'] = pd.to_numeric(df['minutes_used'], errors='coerce')

        # Create the interaction feature
        df['time_platform_interaction'] = df['hour'] * df['minutes_used']

        # Define 3-hour time bins
        df['time_frame'] = (df['hour'] // 3) * 3

        # Prepare numerical and categorical features
        numerical_features = ['time_frame', 'minutes_used', 'time_platform_interaction']
        categorical_features = ['site']

        # Handle categorical features
        site_dummies = pd.get_dummies(df['site'], prefix='site')

        # Combine features
        features = pd.concat([
            df[numerical_features],
            site_dummies
        ], axis=1)

        # Scale numerical features if scaler is available
        if scaler is not None:
            features[numerical_features] = scaler.transform(features[numerical_features])

        return features
    except Exception as e:
        print(f"Error processing data: {e}")
        return None

def get_severity_level(minutes_used):
    """Determine severity level based on minutes used"""
    try:
        minutes_used = float(minutes_used)
        if minutes_used <= 30:
            return 'Low'
        elif minutes_used <= 60:
            return 'Moderate'
        elif minutes_used <= 120:
            return 'High'
        else:
            return 'Critical'
    except:
        return 'Unknown'

@app.route('/screen_time_analysis_dashboard')  # Changed route name to avoid conflict
def screen_time_analysis_dashboard():
    """Route for screen time analysis dashboard"""
    if 'user_id' not in session:
        return redirect('/login')
    return render_template('screen_time_analysis.html')

@app.route('/get_screen_time_predictions')
def get_screen_time_predictions():
    """API endpoint for getting screen time predictions"""
    try:
        # Read the current user's data
        df = pd.read_csv('timedata.csv')
        
        if df.empty:
            return jsonify({
                'success': False,
                'error': 'No data available'
            })
        
        # Process the data
        processed_data = process_data(df)
        
        if processed_data is None:
            return jsonify({
                'success': False,
                'error': 'Error processing data'
            })
        
        # Make predictions
        results = []
        for idx, group in df.groupby('time_frame'):
            total_minutes_used = group['minutes_used'].sum()
            severity = get_severity_level(total_minutes_used)

            results.append({
                'time_frame': f"{int(group['time_frame'].iloc[0]):02d}:00 - {int(group['time_frame'].iloc[0]) + 3:02d}:00",
                'site': group['site'].iloc[0],  # Use the first site's name
                'minutes_used': int(total_minutes_used),
                'severity': severity
            })

        # Calculate summary statistics
        summary = {
            'total_time': int(df['minutes_used'].sum()),
            'critical_count': len([r for r in results if r['severity'] == 'Critical']),
            'high_count': len([r for r in results if r['severity'] == 'High']),
            'moderate_count': len([r for r in results if r['severity'] == 'Moderate']),
            'low_count': len([r for r in results if r['severity'] == 'Low'])
        }
        
        return jsonify({
            'success': True,
            'results': results,
            'summary': summary
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
    



def update_historical_data(user_id):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    # Fetch yesterday's data
    cursor.execute("""
        SELECT hour, minutes_used 
        FROM new_data 
        WHERE user_id = %s AND day = 1
    """, (user_id,))
    yesterday_data = cursor.fetchall()

    # Update historical average data
    for row in yesterday_data:
        hour = row['hour']
        minutes_used = row['minutes_used']
        
        # Check if there is already historical data for the same hour
        cursor.execute("""
            SELECT avg_minutes_used 
            FROM historical_data 
            WHERE user_id = %s AND hour = %s
        """, (user_id, hour))
        historical_row = cursor.fetchone()

        if historical_row:
            # Update the average
            new_avg = (historical_row['avg_minutes_used'] + minutes_used) / 2
            cursor.execute("""
                UPDATE historical_data 
                SET avg_minutes_used = %s 
                WHERE user_id = %s AND hour = %s
            """, (new_avg, user_id, hour))
        else:
            # Insert new historical data
            cursor.execute("""
                INSERT INTO historical_data (user_id, hour, avg_minutes_used)
                VALUES (%s, %s, %s)
            """, (user_id, hour, minutes_used))

    conn.commit()
    cursor.close()
    conn.close()

# Function to move today's data to yesterday
def move_today_to_yesterday(user_id):
    conn = get_db_connection()
    cursor = conn.cursor()

    # Move today's data (day=0) to yesterday (day=1)
    cursor.execute("""
        UPDATE new_data 
        SET day = 1 
        WHERE user_id = %s AND day = 0
    """, (user_id,))

    conn.commit()
    cursor.close()
    conn.close()

# Function to add new day's data
def add_new_day_data(user_id, hour, minutes_used):
    conn = get_db_connection()
    cursor = conn.cursor()

    # Insert new data for today (day=0)
    cursor.execute("""
        INSERT INTO new_data (user_id, day, hour, minutes_used)
        VALUES (%s, 0, %s, %s)
        ON DUPLICATE KEY UPDATE minutes_used = %s
    """, (user_id, hour, minutes_used, minutes_used))

    conn.commit()
    cursor.close()
    conn.close()

# Route to update screen time data for the new day
@app.route('/update_screen_time', methods=['POST'])
def update_screen_time():
    data = request.json
    user_id = data['user_id']
    new_day_data = data['day_data']  # List of {'hour': X, 'minutes_used': Y} for the new day

    # Update historical data with yesterday's data
    update_historical_data(user_id)

    # Move today's data to yesterday
    move_today_to_yesterday(user_id)

    # Insert new data for today
    for entry in new_day_data:
        add_new_day_data(user_id, entry['hour'], entry['minutes_used'])

    return jsonify({"message": "Screen time data updated successfully"}), 200


def get_user(of_user='', all=False):
    cursor = mysql.connection.cursor()
    if of_user != '':
        cursor.execute("""SELECT name, user_id, photo, background, bio FROM users WHERE user_id = '{}'""".format(of_user))
    elif all:
        cursor.execute("""SELECT name, user_id, photo, background, bio, email, pno FROM users WHERE user_id = '{}'""".format(session['user_id']))
    else:
        cursor.execute("""SELECT name, user_id, photo, background, bio FROM users WHERE user_id = '{}'""".format(session['user_id']))
    user = cursor.fetchall()
    
    return user
def calculate_similarity(text1, text2):
    # Create TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer()
    # Fit and transform the input texts
    tfidf_matrix = tfidf_vectorizer.fit_transform([text1, text2])
    # Calculate cosine similarity between the two TF-IDF matrices
    similarity_score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
    return similarity_score
def get_posts(of_user = ''):
    cursor = mysql.connection.cursor()
    if of_user != '':
        # print("""select name, posts.user_id, raw_post, likes, image, posted_at,photo from posts INNER JOIN users on posts.user_id = users.user_id where posts.user_id = '{}'""".format(of_user))
        cursor.execute("""select name, posts.user_id, raw_post, likes, image, posted_at,photo from posts INNER JOIN users on posts.user_id = users.user_id where posts.user_id = '{}' order by posts.id desc""".format(of_user))
    else:
        # print("""select name, posts.user_id, raw_post, likes, image, posted_at,photo from posts INNER JOIN users on posts.user_id = users.user_id""")
        cursor.execute("""select name, posts.user_id, raw_post, likes, image, posted_at,photo from posts INNER JOIN users on posts.user_id = users.user_id order by posts.id desc""")
    posts = cursor.fetchall()
    return posts

def get_path():
    path = request.root_url.replace('/?','')
    return path

@app.route('/')
def home():
    if 'user_id' in session:
        posts = get_posts()
        path = get_path()
        user = get_user()
        # print(posts)
        return render_template('index.html', posts=posts, user=user, path=path)
    else:
        return redirect('/login')

@app.route('/compose')
def compose():
    if 'user_id' in session:
        path = get_path()
        user = get_user()
        x = datetime.datetime.now()
        date = x.strftime("%B %d, %Y")
        return render_template('compose.html', user=user, date=date, path=path)
    else:
        return redirect('/login')

@app.route('/login')
def login():
    if 'user_id' in session:
        return redirect('/')
    else:
        print(request.root_url)
        return render_template('login.html')

@app.route('/register')
def register():
    if 'user_id' in session:
        return redirect('/')
    else:
        return render_template('register.html')

@app.route('/login_validation', methods=['POST'])
def login_validation():
    email = request.form.get('email')
    password = request.form.get('password')
    cursor = mysql.connection.cursor()
    cursor.execute("""SELECT * FROM users WHERE email LIKE '{}'""".format(email))
    user = cursor.fetchone()

    if user:
        # Verify the hashed password using bcrypt
        if bcrypt.checkpw(password.encode('utf-8'), user[4].encode('utf-8')):
            session['user_id'] = user[5]
            return redirect('/')
        else:
            return redirect('/login')
    else:
        return redirect('/login')

@app.route('/add_user', methods=['POST'])
def add_user():
    name=request.form.get('name')
    email=request.form.get('email')
    pno=request.form.get('pno')
    password=request.form.get('password')

    # Hash the password using bcrypt
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

    user_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
    cursor = mysql.connection.cursor()
    cursor.execute("""INSERT INTO users (name, email, pno, password, user_id) values (%s, %s, %s, %s, %s)""", (name, email, pno, hashed_password, user_id))
    mysql.connection.commit()
    return redirect('/login')


@app.route('/new_post', methods=['POST'])
def new_post():
    if 'user_id' in session:
        post_data=request.form.get('post_data')

        user_id = session['user_id']
        x = datetime.datetime.now()
        created_at = x.strftime("%Y-%m-%d %H:%M:%S") 
        f = request.files['image']
        cursor = mysql.connection.cursor()
        if f.filename is not '':
            filename = os.path.join(app.config['UPLOAD_FOLDER'], session['user_id'] + ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))+secure_filename(f.filename))
            f.save(filename)
            print("""INSERT INTO posts (raw_post, user_id, posted_at, image) values ("{}","{}","{}","{}")""".format(post_data, user_id, created_at, filename))
            cursor.execute("""INSERT INTO posts (raw_post, user_id, posted_at, image) values ("{}","{}","{}","{}")""".format(post_data, user_id, created_at, filename))
        else:
            print("""INSERT INTO posts (raw_post, user_id, posted_at) values ("{}","{}","{}")""".format(post_data, user_id, created_at))
            cursor.execute("""INSERT INTO posts (raw_post, user_id, posted_at) values ("{}","{}","{}")""".format(post_data, user_id, created_at))
        mysql.connection.commit()
        return redirect('/')
    else:
        return render_template('login.html')

@app.route('/update-profile', methods=['POST'])
def update_profile():
    if 'user_id' in session:
        name=request.form.get('name')
        email=request.form.get('email')
        pno=request.form.get('pno')
        bio=request.form.get('bio')
        user_id = session['user_id']
        cursor = mysql.connection.cursor()
        # print("""INSERT INTO posts (raw_post, user_id, posted_at) values ("{}","{}","{}")""".format(post_data, user_id, created_at))
        cursor.execute("""UPDATE users SET name = '{}', email = '{}', pno='{}',bio='{}' where user_id = '{}' """.format(name, email, pno, bio, user_id))
        mysql.connection.commit()
        return redirect('/')
    else:
        return redirect('/login')

@app.route('/@<path:user_id>')
def user(user_id):
    if 'user_id' in session:
        path = get_path()
        res = get_user(user_id)
        user = get_user()
        print(res)
        if len(res) > 0:
            posts = get_posts(res[0][1])
            return render_template('profile.html', res = res, user = user, posts = posts, path=path)
        else:
            return f'No User Found'
    else:
        return redirect('/login')

@app.route('/@<path:user_id>/edit-profile')
def edit_profile(user_id):
    if 'user_id' in session:
        if session['user_id'] == user_id:
            path = get_path()
            user = get_user(all=True)
            return render_template('edit-profile.html', user = user, path=path)
        else:
            red = '/@'+user_id
            return redirect(red)
    else:
        return redirect('/login')



@app.route('/logout')
def logout():
    if 'user_id' in session:
        session.pop('user_id')
    return redirect('/login')

@app.route('/test')
def test():
    return render_template('test.html')







@app.route('/select_preference', methods=['GET', 'POST'])
def select_preference():
    if request.method == 'POST':
        preference = request.form['preference']
        session['preference'] = preference
        return redirect(url_for('nearby_places'))
    return render_template('select_preference.html')

@app.route('/nearby_places')
def nearby_places():
    preference = session.get('preference')
    # Initialize 'points' key in session if not already present
    session.setdefault('points', 0)
    return render_template('nearby_places.html', preference=preference)

@app.route('/points')
def points():
    # Initialize 'points' key in session if not already present
    session.setdefault('points', 0)
    return render_template('points.html')

@app.route('/increment_points', methods=['POST'])
def increment_points():
    data = request.json
    points_to_increment = data.get('points', 0)
    session['points'] += points_to_increment
    return jsonify({'success': True})



@app.route('/update_points', methods=['POST'])
def update_points():
    place_name = request.json.get('placeName')
    distance = request.json.get('distance')

    # Retrieve user ID from session or request
    user_id = session.get('user_id')  # Assuming user ID is stored in session

    # Check if the user is authenticated
    if user_id:
        # Query the database to get user's current points
        cursor = mysql.connection.cursor()
        # Fetch the user data from the database
        cursor.execute("SELECT points FROM users WHERE id = %s", (user_id,))
        user_data = cursor.fetchone()

        if user_data:
            current_points = user_data[0]  # Access the first element of the tuple
    # Add your logic here to update the points

            # Define the threshold for the condition
            threshold = 500  # Adjust as needed
            # Compare the distance with the threshold
            if distance <= threshold:
                # Increment points by 10
                new_points = current_points + 10
                # Update user's points in the database
                cursor.execute("UPDATE users SET points = %s WHERE user_id = %s", (new_points, user_id))
                mysql.connection.commit()
                return jsonify(message="Congratulations! You gained 10 points.")
            else:
                return jsonify(message="Sorry, you gained 0 points.")
        else:
            return jsonify(error="User not found.")
    else:
        return jsonify(error="User not authenticated.")
@app.route('/dashboard')
def dashboard():
    cursor = mysql.connection.cursor()
    cursor.execute("SELECT name, points FROM users ORDER BY points DESC")
    users = cursor.fetchall()
    
    # Convert tuple results to dictionaries
    users_dict = []
    for user in users:
        user_dict = {'name': user[0], 'points': user[1]}
        users_dict.append(user_dict)
    
    return render_template('dashboard.html', users=users_dict)

@app.route('/visit_place', methods=['POST'])
def visit_place():
    if 'user_id' in session:
        cursor = mysql.connection.cursor()
        cursor.execute("SELECT * FROM users WHERE user_id = %s", (session['user_id'],))
        user = cursor.fetchone()
        # Simulate condition for demonstration purposes
        condition_satisfied = True
        if condition_satisfied:
            cursor.execute("UPDATE users SET points = points + 10 WHERE user_id = %s", (session['user_id'],))
            mysql.connection.commit()
            return "Points incremented by 10"
        else:
            return "Condition not satisfied, no points added"
    else:
        return redirect(url_for('login'))

@app.route('/fetch_points')
def fetch_points():
    if 'user_id' in session:
        cursor = mysql.connection.cursor()
        cursor.execute("SELECT points FROM users WHERE user_id = %s", (session['user_id'],))
        user_data = cursor.fetchone()
        if user_data:
            points = user_data[0]
            return {'points': points}
        else:
            return {'points': 0}
    else:
        return {'points': 0}

@app.route('/unlock')
def unlock():
    return redirect(url_for('vouchers'))

@app.route('/vouchers')
def vouchers():
    if 'user_id' in session:
        cursor = mysql.connection.cursor()
        cursor.execute("SELECT points FROM users WHERE user_id = %s", (session['user_id'],))
        user_data = cursor.fetchone()
        if user_data:
            points = user_data[0]
        else:
            points = 0
        cursor.close()
    else:
        points = 0
    
    return render_template('vouchers.html', points=points)




@app.route('/chat')
def chat():
    if 'user_id' in session:
        user = get_user()
        if user:
            return render_template('chat.html', user=user)  # Pass the entire user object
        else:
            return render_template('chat.html', user=None)  # Pass None if user not found
    else:
        return redirect('/login')

@socketio.on('connect')
def handle_connect():
    emit('load_messages', messages, broadcast=False)

@socketio.on('send_message')
def handle_message(data):
    user = get_user()  # Assuming get_user() returns the current user's information
    message_data = {'user': user[0][0], 'message': data['message']}  # Assuming user[0] contains the user's name
    emit('receive_message', message_data, broadcast=True)
    messages.append(message_data)
'''
@socketio.on('find_buddy')
def find_buddy():
    # Get the current user's message
    current_user_message = [message['message'] for message in messages if message['user'] == session.get('user_id')]
    if not current_user_message:
        return  # No messages found for the current user

    # Compare the current user's message with other users' messages and find similar users
    similar_users = []
    for message in messages:
        if message['user'] != session.get('user_id'):
            similarity_score = calculate_similarity(current_user_message[0], message['message'])
            if similarity_score > 0.7:  # Adjust the threshold as needed
                similar_users.append(message['user'])

    # Emit similar users to the client
    emit('similar_users', similar_users, broadcast=False)
from flask import jsonify
'''
@app.route('/findbuddy')
def find_buddy():
    current_user_id = session.get('user_id')
    if current_user_id is None:
        return jsonify({'error': 'User not logged in'}), 401

    current_user_message = next((message['message'] for message in messages if message['user'] == current_user_id), None)
    if current_user_message is None:
        return jsonify({'error': 'No messages found for the current user'}), 404

    similar_users = []
    for message in messages:
        if message['user'] != current_user_id:
            similarity_score = calculate_similarity(current_user_message, message['message'])
            if similarity_score > 0.7:  # Adjust the threshold as needed
                similar_users.append(message['user'])

    # Print similar users to console
    print("Similar users:", similar_users)

    # Return similar users as JSON response
    return jsonify(similar_users)



@app.route('/update_time_spent', methods=['POST'])
def update_time_spent():
    time_spent = request.json['timeSpent']
    cur = mysql.connection.cursor()
    cur.execute("INSERT INTO chrome_time_spent (time_spent) VALUES (%s)", (time_spent,))
    mysql.connection.commit()
    cur.close()
    return jsonify({'message': 'Time spent updated successfully'})

@app.route('/time_spent')
def display_time_spent():
    cur = mysql.connection.cursor()
    cur.execute("SELECT max(time_spent) FROM chrome_time_spent")
    total_time_spent_seconds = cur.fetchone()[0]
    cur.close()
    
    # Convert seconds to hours and minutes
    hours = total_time_spent_seconds // 3600
    minutes = (total_time_spent_seconds % 3600) // 60
    
    if hours < 2:
        cursor = mysql.connection.cursor()
        cursor.execute("UPDATE users SET points = points + 10 WHERE user_id = %s", (session['user_id'],))
        mysql.connection.commit()
        cursor.close()
        return jsonify(total_time_spent={'hours': hours, 'minutes': minutes}, message="Points incremented by 10")
    else:
        return jsonify(total_time_spent={'hours': hours, 'minutes': minutes})
    

@app.route('/recommendation')
def recommendation():
    return render_template('recommendation.html')

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(df):
    selected_features = ['title', 'authors', 'categories', 'published_year']
    for feature in selected_features:
        df[feature] = df[feature].fillna('')
    return df

def prepare_model(df):
    combined_features = df['title'] + ' ' + df['categories'] + ' ' + df['authors'] + ' ' + f"{df['published_year']}"
    vectorizer = TfidfVectorizer()
    feature_vectors = vectorizer.fit_transform(combined_features)
    similarity = cosine_similarity(feature_vectors, feature_vectors)
    return similarity
prepared = False

@app.before_request
def before_request():
    global prepared
    if not prepared:
        prepare()
        prepared = True

def prepare():
    global data, similarity
    data = load_data("data.csv")

    data = preprocess_data(data)
    print(data)
    similarity = prepare_model(data)

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        book_name = request.json.get('book_name')
        if not book_name:
            return jsonify({'error': 'No book name provided'}), 400

        list_of_all_titles = data['title'].tolist()
        find_close_match = difflib.get_close_matches(book_name, list_of_all_titles)
        if not find_close_match:
            return jsonify({'error': 'No match found for the given book name'}), 404

        close_match = find_close_match[0]
        index_of_the_book = data[data.title == close_match].index[0]
        similarity_score = list(enumerate(similarity[index_of_the_book]))
        sorted_similar_books = sorted(similarity_score, key=lambda x: x[1], reverse=True)
        recommendations = [data.iloc[book[0]]['title'] for book in sorted_similar_books[:5]]

        return jsonify({'recommendations': recommendations}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/screen_time', methods=['GET'])
def screen_time():
    user_id = request.args.get('user_id')  # Assuming user_id is passed as a query parameter

    # Get screen time data from the database
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    query = """
        SELECT hour, minutes_used 
        FROM time_app 
        WHERE user_id = %s and day=0
        ORDER BY hour ASC
    """
    cursor.execute(query, (user_id,))

    # Fetch all rows
    screen_time_data = cursor.fetchall()

    cursor.close()
    conn.close()

    # Create a CSV response
    output = io.StringIO()  # Create an in-memory string buffer
    writer = csv.writer(output)

    # Write CSV headers
    writer.writerow(['Hour', 'Minutes Used'])

    # Write data rows
    for row in screen_time_data:
        writer.writerow([row['hour'], row['minutes_used']])

    # Move the pointer to the start of the stream
    output.seek(0)

    # Return the CSV data as a response with content type 'text/csv'
    return Response(output, mimetype='text/csv',
                    headers={"Content-Disposition": "attachment;filename=screen_time.csv"})

##screen_time oldcode    
# @app.route('/screen_time')
# def screen_time():
#     # Simulating screen time data (in minutes)
#     screen_time_data = [
#         {"hour": i, "minutes": np.random.randint(0, 60)} for i in range(24)
#     ]
    
#     # Return the data as JSON
#     return jsonify(screen_time_data)
@app.route('/screen_time_area_chart')
def screen_time_area_chart():
    return render_template('screen_time_area_chart.html')

@app.route('/screen_time_analysis')
def screen_time_analysis():
    if 'user_id' in session:
        cursor = mysql.connection.cursor()
        # Get hourly screen time data
        cursor.execute("""
            SELECT 
                HOUR(FROM_UNIXTIME(time_spent)) as hour,
                AVG(time_spent) as avg_time
            FROM chrome_time_spent 
            GROUP BY HOUR(FROM_UNIXTIME(time_spent))
            ORDER BY hour
        """)
        screen_time_data = cursor.fetchall()
        cursor.close()
        
        # Format data for the frontend
        formatted_data = []
        for hour, avg_time in screen_time_data:
            formatted_data.append({
                'hour': hour,
                'screenTime': round(float(avg_time) / 60)  # Convert seconds to minutes
            })
            
        return render_template('screen_time.html', data=formatted_data)
    else:
        return redirect('/login')
    
@app.route('/screen_time_chart')
def screen_time_chart():
    return render_template('screen_time_chart.html')




if __name__ == '__main__':
    socketio.run(app, debug=True)
    load_models()  # Load models when the app starts
    app.run(debug=True)








