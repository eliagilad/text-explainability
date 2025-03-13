import sys
import os

# Define the path to your external repositories
external_repos_path = '../external_repos/'

# List the repositories you want to include
repos = [
    'eraserbenchmark-master/rationale_benchmark',
]

# Add each repository to sys.path
for repo in repos:
    repo_path = os.path.join(external_repos_path, repo)
    if os.path.isdir(repo_path):
        sys.path.append(repo_path)
    else:
        print(f"Warning: {repo_path} does not exist.")
