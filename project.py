# project.py


import pandas as pd
import numpy as np
from pathlib import Path

import plotly.express as px


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def get_assignment_names(grades):
    cats = ['lab', 'project', 'midterm', 'final', 'disc', 'checkpoint']
    assignments = {}
    for cat in cats:
        assignments[cat] = list(filter(lambda x: 
                                       (cat in x.lower() 
                                        and 'Max' not in x 
                                        and 'Lateness' not in x
                                        and 'free_response' not in x
                                        and 'checkpoint' not in x), list(grades.columns)))
    
    assignments['checkpoint'] = list(filter(lambda x: ('checkpoint' in x
                                                        and 'Max' not in x 
                                                        and 'Lateness' not in x), list(grades.columns)))

    return assignments

# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def projects_total(grades):
    grades = grades.copy()

    projects = get_assignment_names(grades)['project']

    for project in projects:
        grades[project] = grades[project].fillna(0)
 
        fr = project + '_free_response'
        if fr not in grades.columns:
            grades[project + 'percentage'] = grades[project] / grades[project + ' - Max Points']
        else:
            grades[fr] = grades[fr].fillna(0)
            grades[project + 'percentage'] = (grades[project] + grades[fr]) / (grades[project + ' - Max Points'] + grades[fr + ' - Max Points'])
    
    n = len(projects)
    return grades.iloc[:, -1 * n:].mean(axis=1)
        


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def lateness_penalty(col):
    def lateness(time):
        time = time.split(':')

        time  = int(time[0]) * 3600 + int(time[1]) * 60 + int(time[2])
        if time <= 60*60*2:
            return 1.0
        if time <= 60 * 60 * 24 * 7:
            return 0.9
        if time <= 60 * 60 * 24 * 7 * 2:
            return 0.7
        return 0.4

    return col.apply(lateness)



# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def process_labs(grades):
    grades = grades.copy()

    labs = get_assignment_names(grades)['lab']

    for lab in labs:
        grades[lab] = grades[lab].fillna(0)
        grades[lab] = grades[lab] * lateness_penalty(grades[lab + ' - Lateness (H:M:S)']) / grades[lab + ' - Max Points']
    
    return grades[labs]


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


def lab_total(processed):
    def calculate_lab_total(labs):
        labs = np.array(labs)
        labs = np.delete(labs, np.argmin(labs))
        return np.sum(labs) / len(labs)
    
    return processed.apply(calculate_lab_total, axis=1)



# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


def total_points(grades):
    grades = grades.copy()

    def get_grades(category):
        assignments = get_assignment_names(grades)[category]
        for assignment in assignments:
            grades[assignment] = grades[assignment].fillna(0)
            grades[assignment] = (grades[assignment]) / (grades[assignment + ' - Max Points'])

    def calculate_total(row):
        row = np.array(row)
        return np.sum(row) / len(row)

    get_grades('disc')
    get_grades('checkpoint')

    discussions = grades[get_assignment_names(grades)['disc']]
    checkpoints = grades[get_assignment_names(grades)['checkpoint']]

    grades['discussion_grade'] = discussions.apply(calculate_total, axis=1)
    grades['checkpoint_grade'] = checkpoints.apply(calculate_total, axis=1)

    def calculate_test(exam):
        tests = get_assignment_names(grades)[exam]

        for test in tests:
            grades[test] = grades[test].fillna(0)
            grades[test + 'percentage'] = grades[test] / grades[test + ' - Max Points']

        return grades[test + 'percentage']
    
    
    project_total = projects_total(grades)
    labs = lab_total(process_labs(grades))

    
    return (grades['discussion_grade'] * .025) + (grades['checkpoint_grade'] * .025) + (project_total * .3) + (labs * .2) + (.15 * calculate_test('midterm')) + (.3 * calculate_test('final'))




# ---------------------------------------------------------------------
# QUESTION 7
# ---------------------------------------------------------------------


def final_grades(total):
    def letter_grade(grade):
        if grade >= .9:
            return 'A'
        if grade >= .8:
            return 'B'
        if grade >= .7:
            return 'C'
        if grade >= .6:
            return 'D'
        return 'F' 
    
    return total.apply(letter_grade)

def letter_proportions(total):
    total = final_grades(total)

    return total.value_counts(normalize=True).sort_values(ascending=False)


# ---------------------------------------------------------------------
# QUESTION 8
# ---------------------------------------------------------------------


def raw_redemption(final_breakdown, question_numbers):
    final_breakdown = final_breakdown.copy()

    # initializes dictionary {question no: point total}
    question_dict = {}
    for col in final_breakdown.columns:
        if col == 'PID':
            continue
        split = col.split(' ')
        question_no = int(split[1])
        point_total = float(split[2][1:])
        if question_no in question_numbers:  
            question_dict[question_no] = point_total

    # sums points earned
    final_breakdown['sum'] = 0
    for question in question_numbers:
        question_series = final_breakdown.iloc[:, question].astype(float)
        question_series = question_series.fillna(0)
        final_breakdown['sum'] += question_series
    
    final_breakdown['Raw Redemption Score'] = final_breakdown['sum'] / sum(question_dict.values())
    return final_breakdown[['PID', 'Raw Redemption Score']]


    
def combine_grades(grades, raw_redemption_scores):
    return grades.merge(raw_redemption_scores, left_on='PID', right_on='PID') # which join?


# ---------------------------------------------------------------------
# QUESTION 9
# ---------------------------------------------------------------------


def z_score(ser):
    return (ser - ser.mean()) / ser.std(ddof = 0)
    
def add_post_redemption(grades_combined):
    grades_combined = grades_combined.copy()

    test = 'Midterm'
    grades_combined[test] = grades_combined[test].fillna(0)
    grades_combined['Midterm Score Pre-Redemption'] = grades_combined[test] / grades_combined[test + ' - Max Points']
    midterms_pre_z = z_score(grades_combined['Midterm Score Pre-Redemption'])
    grades_combined['Midterm Score Pre-Redemption_z'] = midterms_pre_z
    midterms_post_z = z_score(grades_combined['Raw Redemption Score'])
    grades_combined['Midterm Score Post-Redemption_z'] = midterms_post_z


    class_mean = grades_combined['Midterm Score Pre-Redemption'].mean()
    class_std = grades_combined['Midterm Score Pre-Redemption'].std(ddof = 0)
    grades_combined["Midterm Score Post-Redemption"] = np.where(midterms_pre_z < midterms_post_z, (midterms_post_z * class_std) + class_mean, grades_combined['Midterm Score Pre-Redemption'])


    return grades_combined.drop(columns=['Midterm Score Pre-Redemption_z', 'Midterm Score Post-Redemption_z'])
    

    
    


# ---------------------------------------------------------------------
# QUESTION 10
# ---------------------------------------------------------------------

def total_points_post_redemption(grades_combined):
    df_with_redemption = add_post_redemption(grades_combined)
    pre_redemption_grades_series = total_points(grades_combined)

    grades_without_midterm_series = pre_redemption_grades_series - df_with_redemption['Midterm Score Pre-Redemption'] * 0.15
    return grades_without_midterm_series + df_with_redemption['Midterm Score Post-Redemption'] * 0.15
    
        
def proportion_improved(grades_combined):
    return np.count_nonzero(final_grades(total_points(grades_combined)) != final_grades(total_points_post_redemption(grades_combined))) / grades_combined.shape[0]


# ---------------------------------------------------------------------
# QUESTION 11
# ---------------------------------------------------------------------


def section_most_improved(grades_analysis):
    ...
    
def top_sections(grades_analysis, t, n):
    ...


# ---------------------------------------------------------------------
# QUESTION 12
# ---------------------------------------------------------------------


def rank_by_section(grades_analysis):
    ...







# ---------------------------------------------------------------------
# QUESTION 13
# ---------------------------------------------------------------------


def letter_grade_heat_map(grades_analysis):
    ...
