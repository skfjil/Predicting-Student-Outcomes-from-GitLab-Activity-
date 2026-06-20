from flask import Flask, render_template, url_for, redirect, flash, request, jsonify, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, SelectField
from wtforms.validators import InputRequired, Length, ValidationError, Email, DataRequired
from flask_bcrypt import Bcrypt
import os
import torch
import torch.nn as nn
from itsdangerous import URLSafeTimedSerializer
from flask_mail import Mail, Message
import ast
from gitlab import Gitlab
from datetime import datetime, timedelta
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import unicodedata
from flask_migrate import Migrate
from collections import Counter
from typing import List, Dict

def fetch_gitlab_merge_requests(project):
    merge_requests = project.mergerequests.list(state='all', get_all=True)
    return [mr.attributes for mr in merge_requests]

def fetch_gitlab_issues(project):
    issues = project.issues.list(state='all', get_all=True)
    full_issues = []
    
    for issue in issues:
        notes = issue.notes.list(get_all=True)
        full_issues.append({
            **issue.attributes,
            'notes': [note.attributes for note in notes]
        })
    return full_issues

def fetch_gitlab_commits(project):
    commits = project.commits.list(all=True, get_all=True)
    commits_data = []

    for commit in commits:
        detailed_commit = project.commits.get(commit.id)
        commits_data.append({
            "commit_id": detailed_commit.id,
            "author_name": detailed_commit.author_name,
            "author_email": detailed_commit.author_email,
            "created_at": detailed_commit.created_at,
            "lines_added": detailed_commit.stats['additions'],
            "lines_deleted": detailed_commit.stats['deletions'],
            "lines_changed": detailed_commit.stats['additions'] + detailed_commit.stats['deletions'],
        })
    return commits_data

def get_repositories_info(token):
    try:
        GITLAB_URL = 'https://gitlab.com'
        gl = Gitlab(GITLAB_URL, private_token=token)

        gl.auth()
        repositories_list = []

        for project in gl.projects.list(membership=True, all=True):
            id_and_name = [project.id, project.name]
            repositories_list.append(id_and_name)

        return repositories_list
    except Exception as e:
        print(f"Error fetching repositories: {e}")
        return None

from gitlab.exceptions import GitlabAuthenticationError, GitlabGetError, GitlabHttpError

def get_repositories_info_name(token, name):
    repositories_list = []
    GITLAB_URL = 'https://gitlab.com'

    try:
        gl = Gitlab(GITLAB_URL, private_token=token)
        gl.auth()  # explicitly test authentication
    except GitlabAuthenticationError:
        print("Unauthorized: Invalid or expired token.")
        return []
    except GitlabHttpError as e:
        print(f"HTTP error during authentication: {e}")
        return []

    try:
        projects = gl.projects.list(search=name, per_page=5)
        for project in projects:
            repositories_list.append([project.id, project.name])
    except GitlabGetError as e:
        print(f"Error fetching projects list: {e}")
        return []

    try:
        project = gl.projects.get(name)
        repositories_list.append([project.id, project.name])
    except GitlabGetError as e:
        print(f"Error fetching specific project: {e}")
    except Exception as e:
        print(f"Unexpected error fetching specific project: {e}")

    return repositories_list


def normalize_name(username):
    username = username.lower()
    username = username.translate(str.maketrans('','', "@ .-_"))
    username = ''.join(
        c for c in unicodedata.normalize('NFKD', username)
        if not unicodedata.combining(c)
    )
    return username

def initiate_user(all_data, username, team_name):
    all_data[username] = {
        'username': username,
        'team_name': team_name,
        'number_of_merges_created': 0,
        'number_of_merges_merged': 0,
        'lines_added': 0,
        'lines_deleted': 0,
        'lines_changed': 0,
        'number_of_commits': 0,
        'number_of_issues_created': 0,
        'number_of_issues_assigned': 0,
        'number_of_issues_closed': 0,
        # Date tracking fields
        'commit_dates': [],
        'mr_created_dates': [],
        'mr_merged_dates': [],
        'issue_created_dates': [],
        'issue_closed_dates': []
    }

def get_all_information(id, name, token):
    GITLAB_URL = 'https://gitlab.com'
    gl = Gitlab(GITLAB_URL, private_token=token)

    project = gl.projects.get(id)
    project_info = {"id": id, "name": name}

    project_data = {
        "project_info": project_info,
        "merge_requests": fetch_gitlab_merge_requests(project),
        "issues": fetch_gitlab_issues(project),
        "commits": fetch_gitlab_commits(project)
    }
    
    return project_data

def extract_features(project_data):
    all_data = {}
    team_name = project_data['project_info']['name']

    for issue in project_data['issues']:
        username = normalize_name(issue['author']['username'])
        if username not in all_data:
            initiate_user(all_data, username, team_name)
        all_data[username]['number_of_issues_created'] += 1
        all_data[username]['issue_created_dates'].append(issue['created_at'])

        for assignee in issue.get('assignees', []):
            assignee_username = normalize_name(assignee['username'])
            if assignee_username not in all_data:
                initiate_user(all_data, assignee_username, team_name)
            all_data[assignee_username]['number_of_issues_assigned'] += 1

        if issue['state'] == 'closed' and issue.get('closed_by'):
            closer_username = normalize_name(issue['closed_by']['username'])
            if closer_username not in all_data:
                initiate_user(all_data, closer_username, team_name)
            all_data[closer_username]['number_of_issues_closed'] += 1
            all_data[closer_username]['issue_closed_dates'].append(issue.get('closed_at'))

    for mr in project_data['merge_requests']:
        creator_username = normalize_name(mr['author']['username'])
        if creator_username not in all_data:
            initiate_user(all_data, creator_username, team_name)
        all_data[creator_username]['number_of_merges_created'] += 1
        all_data[creator_username]['mr_created_dates'].append(mr['created_at'])

        if mr['state'] == 'merged' and mr.get('merged_by'):
            merger_username = normalize_name(mr['merged_by']['username'])
            if merger_username not in all_data:
                initiate_user(all_data, merger_username, team_name)
            all_data[merger_username]['number_of_merges_merged'] += 1
            all_data[merger_username]['mr_merged_dates'].append(mr.get('merged_at'))
    
    
    """USERNAME TO NAME MAPPING"""
        # Build map from normalized author_name (from issues/MRs) to username
    name_to_username = {}

    for issue in project_data['issues']:
        norm_name = normalize_name(issue['author']['name'])
        norm_username = normalize_name(issue['author']['username'])
        name_to_username[norm_name]     = norm_username
        name_to_username[norm_username] = norm_username  # map handle→itself

    for mr in project_data['merge_requests']:
        norm_name = normalize_name(mr['author']['name'])
        norm_username = normalize_name(mr['author']['username'])
        name_to_username[norm_name]     = norm_username
        name_to_username[norm_username] = norm_username


    """"""
    for commit in project_data['commits']:
        #username = normalize_name(commit['author_name'])

        norm_commit_name  = normalize_name(commit['author_name'])
        norm_email_part   = normalize_name(commit['author_email'].split('@')[0])

        username = (
            name_to_username.get(norm_commit_name)
            or name_to_username.get(norm_email_part)
            or norm_commit_name
        )

        
        if username not in all_data:
            initiate_user(all_data, username, team_name)
        all_data[username]['lines_added'] += commit['lines_added']
        all_data[username]['lines_deleted'] += commit['lines_deleted']
        all_data[username]['lines_changed'] += commit['lines_changed']
        all_data[username]['number_of_commits'] += 1
        all_data[username]['commit_dates'].append(commit['created_at'])

    return pd.DataFrame(all_data).T

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = "sqlite:///" + os.path.join(os.path.abspath(os.path.dirname(__file__)), "database.db")
db = SQLAlchemy(app)
app.config['SECRET_KEY'] = 'thisisasecretkey'
bcrypt = Bcrypt(app)

app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'matasbazz@gmail.com'
app.config['MAIL_PASSWORD'] = 'fgoa uizt dpak oocy'
app.config['MAIL_DEFAULT_SENDER'] = 'matasbazz@gmail.com'

mail = Mail(app)
serializer = URLSafeTimedSerializer(app.config['SECRET_KEY'])

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

migrate = Migrate(app, db)

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(9, 64)
        self.fc2 = nn.Linear(64, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x)) 
        x = self.fc2(x)
        return x
    
model = MyModel()

model.load_state_dict(torch.load(r'Code\model_new.pth'))

model.eval()

mean = torch.tensor([[5.9453e+00, 2.5469e+00, 1.0781e+05, 1.4591e+04, 1.2240e+05, 4.5195e+01,
         1.6883e+01, 2.2375e+01, 1.0703e+01]])
std = torch.tensor([[6.5752e+00, 6.7636e+00, 8.4143e+05, 7.7710e+04, 8.4790e+05, 4.0953e+01,
         1.6893e+01, 1.4977e+01, 1.7163e+01]])

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class Repository(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key = True)
    name = db.Column(db.String(20))


class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key = True)
    #username = db.Column(db.String(20), nullable=False, unique = True)
    password = db.Column(db.String(80), nullable=False)
    email = db.Column(db.String(120), nullable=False, unique=True)
    verified = db.Column(db.Boolean, default=False)
    occupation = db.Column(db.String(10), nullable=False)

class Student(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20))
    repository_id = db.Column(db.Integer, db.ForeignKey('repository.id'))
    
    number_of_merges_created = db.Column(db.Integer)
    number_of_merges_merged = db.Column(db.Integer)
    
    lines_added = db.Column(db.Integer)
    lines_deleted = db.Column(db.Integer)
    lines_changed = db.Column(db.Integer)
    
    number_of_commits = db.Column(db.Integer)
    
    number_of_issues_created = db.Column(db.Integer)
    number_of_issues_assigned = db.Column(db.Integer)
    number_of_issues_closed = db.Column(db.Integer)
    grade_prediction = db.Column(db.Integer, nullable=True)

    commit_dates = db.Column(db.JSON) 
    mr_created_dates = db.Column(db.JSON)
    mr_merged_dates = db.Column(db.JSON)
    issue_created_dates = db.Column(db.JSON)
    issue_closed_dates = db.Column(db.JSON)

    

class RegisterForm(FlaskForm):
    #username = StringField(validators = [InputRequired(), Length(
    #    min = 4, max = 20)], render_kw={"placeholder" : "Username"})
    email = StringField(validators=[InputRequired(), Email()], render_kw={"placeholder": "Email"})
    password = PasswordField(validators = [InputRequired(), Length(
        min = 4, max = 20)], render_kw={"placeholder" : "Password"})
    occupation = SelectField("Occupation", choices=[("Professor", "Professor"), ("Student", "Student")], validators=[InputRequired()])

    submit = SubmitField("Register")
    """
    def validate_username(self, username):
        existing_user_username = User.query.filter_by(
            username = username.data).first()
        if existing_user_username:
            raise ValidationError(
                "That username already exits. Please choose a different one."
            )
    """
    def validate_email(self, email):
        existing_email = User.query.filter_by(email=email.data).first()
        if existing_email:
            raise ValidationError("That email is already registered.")
        
class LoginForm(FlaskForm):
    #username = StringField(validators = [InputRequired(), Length(
    #    min = 4, max = 20)], render_kw={"placeholder" : "Username"})
    email = StringField(validators=[InputRequired(), Email()], render_kw={"placeholder": "Email"})
    password = PasswordField(validators = [InputRequired(), Length(
        min = 4, max = 20)], render_kw={"placeholder" : "Password"})
    
    submit = SubmitField("Login")

@app.route('/dashboard', methods = ["GET", "POST"])
@login_required
def dashboard():
    return render_template('dashboard.html', email=current_user.email.split('@')[0])

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/repositories/students/<int:repo_id>', methods=["GET"])
@login_required
def students_for_repo(repo_id):
    selected_repo = Repository.query.get(repo_id)
    students = Student.query.filter_by(repository_id=repo_id).all()
    return render_template('students_for_repo.html', repo=selected_repo, students=students)

@app.route('/repositories', methods=["GET", "POST"])
@login_required
def repositories():
    repositories = Repository.query.all()
    if request.method == 'POST':
        repo_id = request.form.get('repo')
        return redirect(url_for('students_for_repo', repo_id=repo_id))
    return render_template('repositories.html', repositories = repositories)

@app.route('/repositories/students/<int:repo_id>/<int:student_id>', methods=['POST'])
@login_required
def remove_student(repo_id, student_id):
    student = Student.query.get_or_404(student_id)

    try:
        db.session.delete(student)
        db.session.commit()

        # Check if there are remaining students
        students = Student.query.filter_by(repository_id=repo_id).all()
        if not students:
            repo = Repository.query.get(repo_id)
            if repo:
                db.session.delete(repo)
                db.session.commit()
                #flash('Student and empty repository removed.', 'success')
                return redirect(url_for('repositories'))  # Go back to repo list

        #flash('Student removed successfully.', 'success')

    except Exception as e:
        db.session.rollback()
        #flash('Error removing student or repository: {}'.format(str(e)), 'danger')

    return redirect(url_for('students_for_repo', repo_id=repo_id))

@app.route('/repositories/add_new', methods=["GET", "POST"])
@login_required
def repositories_add_new():
    if request.method == 'POST':
        action = request.form.get('action')
        if action == 'submit_token':    
            token = request.form.get('token')        
            if token:
                repositories = get_repositories_info(token)
                session['repositories'] = repositories
                session['gitlab_token'] = token
                return redirect(url_for('repositories_add_new'))
        if action == 'submit_name':
            name = request.form.get('gitlab_name')
            token = request.form.get('token')
            if token and name:
                repositories = get_repositories_info_name(token, name)
                session['repositories'] = repositories
                session['gitlab_token'] = token
                return redirect(url_for('repositories_add_new'))
        repo = request.form.get('repo')
        if repo:
            repo = ast.literal_eval(repo)
            id = int(repo[0])
            name = repo[1]
            token = session.get('gitlab_token')
            full_data = get_all_information(id, name, token)
            extracted_data = extract_features(full_data) 

            repository = Repository.query.filter_by(id=id).first()

            if not repository:
                repository = Repository(
                    id=id,
                    name=name
                )
                db.session.add(repository)
                db.session.commit()
            
            for index, user_data in extracted_data.iterrows():
                student = Student.query.filter_by(username=user_data['username'], repository_id=id).first()
                if not student:
                    student = Student(username=user_data['username'],repository_id=id)
                    db.session.add(student)
                student.number_of_merges_created = user_data['number_of_merges_created']
                student.number_of_merges_merged = user_data['number_of_merges_merged']
                student.lines_added = user_data['lines_added']
                student.lines_deleted = user_data['lines_deleted']
                student.lines_changed = user_data['lines_changed']
                student.number_of_commits = user_data['number_of_commits']
                student.number_of_issues_created = user_data['number_of_issues_created']
                student.number_of_issues_assigned = user_data['number_of_issues_assigned']
                student.number_of_issues_closed = user_data['number_of_issues_closed']
                student.commit_dates = user_data['commit_dates']
                student.mr_created_dates = user_data['mr_created_dates']
                student.mr_merged_dates = user_data['mr_merged_dates']
                student.issue_created_dates = user_data['issue_created_dates']
                student.issue_closed_dates = user_data['issue_closed_dates']
                student.grade_prediction = None
                db.session.commit()
            session.pop('repositories', None)
            return redirect(url_for('students_for_repo', repo_id=id))
    repositories = session.pop('repositories', None)
    return render_template('repositories_add_new.html', repositories=repositories)

"""
@app.route('/repositories/refresh/<int:repo_id>', methods=["POST"])
@login_required
def refresh_repo(repo_id):
    token = session.get('gitlab_token')  # This assumes the token is in the session
    repo = Repository.query.get(repo_id)
    if repo and token:
        try:
            full_data = get_all_information(repo.id, repo.name, token)
            extracted_data = extract_features(full_data)

            # Clear and re-add students
            Student.query.filter_by(repository_id=repo.id).delete()
            for _, user_data in extracted_data.iterrows():
                new_student = Student(
                    username=user_data['username'],
                    repository_id=repo.id,
                    number_of_merges_created=user_data['number_of_merges_created'],
                    number_of_merges_merged=user_data['number_of_merges_merged'],
                    lines_added=user_data['lines_added'],
                    lines_deleted=user_data['lines_deleted'],
                    lines_changed=user_data['lines_changed'],
                    number_of_commits=user_data['number_of_commits'],
                    number_of_issues_created=user_data['number_of_issues_created'],
                    number_of_issues_assigned=user_data['number_of_issues_assigned'],
                    number_of_issues_closed=user_data['number_of_issues_closed']
                )
                db.session.add(new_student)
            db.session.commit()
            flash("Repository data refreshed!", "success")
        except Exception as e:
            flash(f"Error refreshing repository: {str(e)}", "danger")
    else:
        flash("Missing token or repository.", "warning")
    return redirect(url_for('students_for_repo', repo_id=repo_id))
"""
@app.route('/repositories/delete/<int:repo_id>', methods=['POST'])
@login_required
def delete_repository(repo_id):
    repo = Repository.query.get_or_404(repo_id)  # Get the repository or return 404 error

    # Delete associated student data first
    Student.query.filter_by(repository_id=repo_id).delete()
    db.session.delete(repo)  # Delete the repository itself
    db.session.commit()

    #flash(f"Repository '{repo.name}' and associated data deleted successfully!", 'success')
    return jsonify({'result': 'success'})  # Return a JSON response


@app.route('/predictions', methods=["GET", "POST"])
@login_required
def predictions():
    repositories = Repository.query.all()
    repo_student_counts = {
        repo.id: Student.query.filter_by(repository_id=repo.id).count()
        for repo in repositories
    }
    if request.method == 'POST':
        repo_id = request.form.get('repo')
        return redirect(url_for('predictions_for_repo', repo_id=repo_id))
    return render_template('predictions.html', repositories = repositories, repo_student_counts=repo_student_counts)

@app.route('/predictions/students/<int:repo_id>', methods=["GET", "POST"])
@login_required
def predictions_for_repo(repo_id):
    selected_repo = Repository.query.get(repo_id)
    students = Student.query.filter_by(repository_id=repo_id).all()
    selected_student = None
    if request.method == "POST":
        action = request.form.get('action')
        if action == 'predict_one': 
            student_id = request.form.get("student_id")
            if student_id:
                selected_student = Student.query.get(int(student_id))
                if selected_student:
                    features = [
                        selected_student.number_of_merges_created,
                        selected_student.number_of_merges_merged,
                        selected_student.lines_added,
                        selected_student.lines_deleted,
                        selected_student.lines_changed,
                        selected_student.number_of_commits,
                        selected_student.number_of_issues_created,
                        selected_student.number_of_issues_assigned,
                        selected_student.number_of_issues_closed,
                    ]
                    X = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
                    X = (X - mean) / std
                    Y = round(model(X).item())
                    if Y > 20: Y = 20
                    if Y < 0 : Y = 0
                    selected_student.grade_prediction = Y
                    db.session.commit()
        elif action == "predict_all":
            for student in students:
                features = [
                    student.number_of_merges_created,
                    student.number_of_merges_merged,
                    student.lines_added,
                    student.lines_deleted,
                    student.lines_changed,
                    student.number_of_commits,
                    student.number_of_issues_created,
                    student.number_of_issues_assigned,
                    student.number_of_issues_closed,
                ]
                X = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
                X = (X - mean) / std
                Y = round(model(X).item())
                if Y > 20: Y = 20
                if Y < 0 : Y = 0
                student.grade_prediction = Y
            db.session.commit()
            

    return render_template(
        'predictions_for_repo.html',
        repo=selected_repo,
        students=students,
        selected_student=selected_student
    )

@app.route('/student_history', methods=["GET", "POST"])
@login_required
def student_history():
    repositories = Repository.query.all()
    repo_student_counts = {
        repo.id: Student.query.filter_by(repository_id=repo.id).count()
        for repo in repositories
    }
    if request.method == 'POST':
        repo_id = request.form.get('repo')
        return redirect(url_for('student_history_for_repo', repo_id=repo_id))
    return render_template('student_history.html', repositories = repositories, repo_student_counts=repo_student_counts)

@app.route('/student-history/students/<int:repo_id>', methods=["GET"])
@login_required
def student_history_for_repo(repo_id):
    selected_repo = Repository.query.get(repo_id)
    students = Student.query.filter_by(repository_id=repo_id).all()
    return render_template('student_history_for_repo.html', repo=selected_repo, students=students)

def compute_weekly_counts(date_strings: List[str]) -> Dict[str, int]:
    if not date_strings:
        return {}
    dates = []
    for ds in date_strings or []:
        try:
            ds_clean = ds.replace('Z', '+00:00')
            dt = datetime.fromisoformat(ds_clean)
        except ValueError:
            try:
                dt = datetime.strptime(ds, "%Y-%m-%d")
            except:
                continue  
        dates.append(dt)
    if not dates:
        return {}
    min_date = min(dates)
    max_date = max(dates)
    
    min_date = min_date - timedelta(days=min_date.weekday())
    max_date = max_date + timedelta(days=(6 - max_date.weekday()))
    
    current = min_date
    weeks = {}
    while current <= max_date:
        y, w, _ = current.isocalendar()
        week_label = f"{y}-W{w:02d}"
        weeks[week_label] = 0
        current += timedelta(weeks=1)
    
    for dt in dates:
        y, w, _ = dt.isocalendar()
        week_label = f"{y}-W{w:02d}"
        weeks[week_label] += 1

    return weeks

def get_student_weekly_metrics(student) -> Dict[str,Dict[str,int]]:
    return {
        'Commits'         : compute_weekly_counts(student.commit_dates),
        'MRs Created'     : compute_weekly_counts(student.mr_created_dates),
        'MRs Merged'      : compute_weekly_counts(student.mr_merged_dates),
        'Issues Opened'   : compute_weekly_counts(student.issue_created_dates),
        'Issues Closed'   : compute_weekly_counts(student.issue_closed_dates),
    }

@app.route('/student-history/metrics/<int:student_id>')
@login_required
def student_history_metrics(student_id):
    student = Student.query.get_or_404(student_id)
    if not student:
        return jsonify({"error": "Student not found"}), 404
    metrics = get_student_weekly_metrics(student)
    # Check for empty data
    if all(len(m) == 0 for m in metrics.values()):
        return jsonify({"labels": [], "datasets": []})
    all_weeks = sorted(
        {wk for metric in metrics.values() for wk in metric},
        key=lambda w: datetime.strptime(w + "-1", "%Y-W%W-%w")  # Sort by ISO week
    )
    payload = {
        "labels": all_weeks,
        "datasets": [
            {
                "label": name,
                "data": [metric.get(wk, 0) for wk in all_weeks]
            } for name, metric in metrics.items()
        ]
    }
    return jsonify(payload)

def extract_weeks(weekly_metrics):
    week_set = set()
    for metric_data in weekly_metrics.values():
        week_set.update(metric_data.keys())
    return sorted(week_set)

def get_cumulative_metrics_up_to(weekly_metrics, target_week):
    result = {}

    for metric, weeks_data in weekly_metrics.items():
        total = 0
        for week, count in weeks_data.items():
            if week <= target_week:
                total += count
        result[metric] = total

    return result

@app.route('/student-history/comparison/<int:student_id>', methods=['GET', 'POST'])
@login_required
def prediction_comparison(student_id):
    student = Student.query.get_or_404(student_id)
    weekly_metrics = get_student_weekly_metrics(student)
    weeks = extract_weeks(weekly_metrics)

    if request.method == 'POST':
        week1 = request.form.get('week1')
        week2 = request.form.get('week2')

        if not week1 or not week2:
            flash("Please select both weeks for comparison.", "error")
            return redirect(url_for('prediction_comparison', student_id=student_id))

        week1_metrics = get_cumulative_metrics_up_to(weekly_metrics, week1)
        week2_metrics = get_cumulative_metrics_up_to(weekly_metrics, week2)

        X1 = [
            week1_metrics['MRs Created'],
            week1_metrics['MRs Merged'],
            (student.lines_added * week1_metrics['Commits'] / student.number_of_commits) if student.number_of_commits else 0,
            (student.lines_deleted * week1_metrics['Commits'] / student.number_of_commits) if student.number_of_commits else 0,
            (student.lines_changed * week1_metrics['Commits'] / student.number_of_commits) if student.number_of_commits else 0,
            week1_metrics['Commits'],
            week1_metrics['Issues Opened'],
            (student.number_of_issues_assigned * week1_metrics['Issues Opened'] / student.number_of_issues_created) if student.number_of_issues_created else 0,
            week1_metrics['Issues Closed']
        ]
        X1 = torch.tensor(X1, dtype=torch.float32).unsqueeze(0)
        print(X1)
        #X1 = torch.log(X1+1)
        X1 = (X1 - mean) / std
        Y1 = round(model(X1).item())
        Y1 = max(0, min(Y1, 20))

        X2 = [
            week2_metrics['MRs Created'],
            week2_metrics['MRs Merged'],
            (student.lines_added * week2_metrics['Commits'] / student.number_of_commits) if student.number_of_commits else 0,
            (student.lines_deleted * week2_metrics['Commits'] / student.number_of_commits) if student.number_of_commits else 0,
            (student.lines_changed * week2_metrics['Commits'] / student.number_of_commits) if student.number_of_commits else 0,
            week2_metrics['Commits'],
            week2_metrics['Issues Opened'],
            (student.number_of_issues_assigned * week2_metrics['Issues Opened'] / student.number_of_issues_created) if student.number_of_issues_created else 0,
            week2_metrics['Issues Closed']
        ]
        X2 = torch.tensor(X2, dtype=torch.float32).unsqueeze(0)
        print(X2)
        #X2 = torch.log(X2 + 1)
        X2 = (X2 - mean) / std
        Y2 = round(model(X2).item())
        Y2 = max(0, min(Y2, 20))
        print(student.username)
        return render_template(
            'prediction_comparison.html',
            student=student,
            weeks=weeks,
            week1=week1,
            week2=week2,
            y1=Y1,
            y2=Y2
        )

    return render_template('prediction_comparison.html', weeks=weeks, student=student)




@app.route('/logout', methods = ['GET', 'POST'])
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/register', methods=["GET", "POST"])
def register():
    form = RegisterForm()

    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data)
        new_user = User(password=hashed_password, email=form.email.data, verified=False, occupation=form.occupation.data)
        db.session.add(new_user)
        db.session.commit()

        # Generate confirmation token and send email
        token = serializer.dumps(new_user.email, salt='email-confirm')
        confirm_url = url_for('confirm_email', token=token, _external=True)
        msg = Message('Confirm Your Email', 
                      recipients=[new_user.email],
                      sender='smtp@mailtrap.io')
        msg.body = f'Click the link to confirm your email: {confirm_url}'
        mail.send(msg)

        # Flash a message
        flash("A verification email has been sent. Please check your inbox.", "success")
        
        # You can redirect or render the same page if needed
        return redirect(url_for('register'))  # This will show the flash message

    return render_template('register.html', form=form)

@app.route('/confirm_email/<token>')
def confirm_email(token):
    try:
        email = serializer.loads(token, salt="email-confirm", max_age=3600)  # 1-hour expiration
        user = User.query.filter_by(email=email).first()

        if user and not user.verified:
            user.verified = True
            db.session.commit()
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            return "Invalid or already verified token."
    
    except:
        return "The confirmation link is invalid or has expired."


@app.route('/login', methods=["GET", "POST"])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if not user:
            flash("This email is not registered.", "danger")
            return redirect(url_for('login'))
        if not bcrypt.check_password_hash(user.password, form.password.data):
            flash("Incorrect password. Please try again.", "danger")
            return redirect(url_for('login'))
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            if user.verified:
                login_user(user)
                return redirect(url_for('dashboard'))
            else:
                return "Please verify your email before logging in."
    
    return render_template('login.html', form=form)


if __name__ == "__main__":
    app.run(debug=True)
