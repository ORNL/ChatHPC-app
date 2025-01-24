#!/bin/bash

# Prompt for new project name.
echo "Note: Please fill in every field. This script is not smart enought to handle missing entries."
read -p "Enter new project name (i.e. ChatKokkos): " project_name
read -p "Enter new project description. " project_description
read -p "Enter new project slug (i.e. ChatKokkos): " project_slug
read -p "Enter new python project name (i.e. chatkokkos): " project_path
read -p "Enter new project group/user (i.e. ChatHPC): " project_gitlab_path
read -p "Enter new author (i.e. Aaron Young): " author_name
read -p "Enter new author email (i.e. youngar@ornl.gov): " author_email

echo "****** Update Template for project ******"
echo "*** Project Name        = ${project_name}"
echo "*** Project Description = ${project_description}"
echo "*** Project Slug        = ${project_slug}"
echo "*** Project Path        = ${project_path}"
echo "*** Project GitLab Path = ${project_gitlab_path}"
echo "*** Author Name         = ${author_name}"
echo "*** Author Email        = ${author_email}"
echo "*****************************************"

read -p "Continue? (Y/N): " confirm && [[ $confirm == [yY] || $confirm == [yY][eE][sS] ]] || exit 1

FILES=".editorconfig .gitignore .gitlab-ci.yml .pre-commit-config.yaml CHANGELOG.md README.md docs/api.md docs/index.md mkdocs.yml pyproject.toml ruff_defaults.toml scripts/test_commands.sh src/chatkokkos/__about__.py src/chatkokkos/__init__.py src/chatkokkos/add.py tests/__init__.py tests/test_add.py scripts/version_bump.py"

echo Project Name:
echo sed -i "s#ChatKokkos#$project_name#g" $FILES
sed -i "s#ChatKokkos#$project_name#g" $FILES
echo

echo Project Description:
echo sed -i "s/\"ChatHPC project for Kokkos.\"/\"$project_description\"/g" $FILES
sed -i "s/\"ChatHPC project for Kokkos.\"/\"$project_description\"/g" $FILES
echo

echo Project Slug:
echo sed -i "s#ChatKokkos#$project_slug#g" $FILES
sed -i "s#ChatKokkos#$project_slug#g" $FILES
echo

echo Project Path:
echo sed -i "s#chatkokkos#$project_path#g" $FILES
sed -i "s#chatkokkos#$project_path#g" $FILES
echo

echo Project GitLab Path:
echo sed -i "s#ChatHPC#$project_gitlab_path#g" $FILES
sed -i "s#ChatHPC#$project_gitlab_path#g" $FILES
echo

echo Author Name:
echo sed -i "s#Aaron Young#$author_name#g" pyproject.toml
sed -i "s#Aaron Young#$author_name#g" pyproject.toml
echo

echo Author Email:
echo sed -i "s#youngar@ornl.gov#$author_email#g" $FILES
sed -i "s#youngar@ornl.gov#$author_email#g" $FILES
echo

echo Move project source files
echo mv src/chatkokkos src/${project_path}
mv src/chatkokkos src/${project_path}

echo Update setup_template.sh with new project information.
sed -i "s#ChatKokkos#$project_name#g" setup_template.sh
sed -i "s/ChatHPC project for Kokkos./$project_description/g" setup_template.sh
sed -i "s#ChatKokkos#$project_slug#g" setup_template.sh
sed -i "s#chatkokkos#$project_path#g" setup_template.sh
sed -i "s#ChatHPC#$project_gitlab_path#g" setup_template.sh
sed -i "s#Aaron Young#$author_name#g" setup_template.sh
sed -i "s#youngar@ornl.gov#$author_email#g" setup_template.sh

echo
echo '*** Please manually check the changes made by this script using git before committing the changes. ***'
echo
git status
