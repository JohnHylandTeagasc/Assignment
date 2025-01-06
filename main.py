print('hello world')

import os
print("Current Working Directory:", os.getcwd())

os.chdir(r"c:\Users\John.Hyland\OneDrive - Teagasc - Agriculture and Food Development Authority\Documents\University of Vienna\Module 2\Module 2.7\CoursePackage")
print("New Working Directory:", os.getcwd())

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = np.loadtxt(fname='OpenDataSurvey.txt', delimiter='\t', skiprows=1)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = np.loadtxt(fname='OpenDataSurvey.txt', delimiter='\t', skiprows=1)

# Print the loaded data
print("Loaded Data:")
print(data)

# Column ranges for constructs
constructs = {
    "Perceived Behavioral Control": data[:, 0:3],  # PBC1, PBC2, PBC3
    "Perceived Effort": data[:, 3:6],             # PE1, PE2, PE3
    "Perceived Risk": data[:, 6:9],              # PRT1, PRT2, PRT3
    "Attitude": data[:, 9:13],                   # ATT1, ATT2, ATT3, ATT4
    "Social Norms": data[:, 13:16],              # SN1, SN2, SN3
    "Facilitating Conditions": data[:, 16:19],   # FC1, FC2, FC3
    "Intention": data[:, 19:22]                  # INT1, INT2, INT3
}

# Classification function
def classify(score, construct_name):
    """
    Classifies scores based on construct-specific ranges.
    """
    if construct_name == "Attitude":
        if 0 <= score <= 9:
            return "Low"
        elif 10 <= score <= 18:
            return "Medium"
        elif 19 <= score <= 28:
            return "High"
    else:  # For all other constructs
        if 0 <= score <= 7:
            return "Low"
        elif 8 <= score <= 14:
            return "Medium"
        elif 15 <= score <= 21:
            return "High"
    return "Invalid"

# Prepare data for the horizontal bar chart and spider diagram
construct_names = list(constructs.keys())
low_counts, medium_counts, high_counts = [], [], []
low_percentages, medium_percentages, high_percentages = [], [], []

for construct_name, columns in constructs.items():
    # Calculate scores
    scores = np.sum(columns, axis=1)
    
    # Classify scores (pass the construct_name as an argument)
    categories = np.array([classify(score, construct_name) for score in scores])  # FIXED HERE
    
    # Count the number of respondents in each category
    unique_categories, counts = np.unique(categories, return_counts=True)
    counts_dict = dict(zip(unique_categories, counts))
    
    # Get counts for Low, Medium, and High (default to 0 if missing)
    low_count = counts_dict.get("Low", 0)
    medium_count = counts_dict.get("Medium", 0)
    high_count = counts_dict.get("High", 0)
    total_count = low_count + medium_count + high_count
    
    # Append counts for horizontal bar chart
    low_counts.append(low_count)
    medium_counts.append(medium_count)
    high_counts.append(high_count)
    
    # Calculate percentages for spider diagram
    low_percentages.append((low_count / total_count) * 100)
    medium_percentages.append((medium_count / total_count) * 100)
    high_percentages.append((high_count / total_count) * 100)

# Create the horizontal bar chart
plt.figure(figsize=(12, 8))
y_pos = np.arange(len(construct_names))
plt.barh(y_pos - 0.2, low_counts, color="red", height=0.3, label="Low")
plt.barh(y_pos, medium_counts, color="orange", height=0.3, label="Medium")
plt.barh(y_pos + 0.2, high_counts, color="green", height=0.3, label="High")

# Add labels and title
plt.yticks(y_pos, construct_names)
plt.xlabel("Number of Respondents")
plt.title("Number of High, Medium and Low Category Respondents Across Open Data Constructs")
plt.legend(loc="upper right")

# Adjust layout
plt.tight_layout()
plt.show()

# Prepare data for the spider diagram
categories = construct_names + [construct_names[0]]  # Close the circle
low_percentages += [low_percentages[0]]
medium_percentages += [medium_percentages[0]]
high_percentages += [high_percentages[0]]

# Create spider diagram
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=True)

# Plot percentages for Low, Medium, and High
ax.plot(angles, low_percentages, color="red", linewidth=2, linestyle="solid", label="Low")
ax.fill(angles, low_percentages, color="red", alpha=0.25)

ax.plot(angles, medium_percentages, color="orange", linewidth=2, linestyle="solid", label="Medium")
ax.fill(angles, medium_percentages, color="orange", alpha=0.25)

ax.plot(angles, high_percentages, color="green", linewidth=2, linestyle="solid", label="High")
ax.fill(angles, high_percentages, color="green", alpha=0.25)

# Add labels and title
ax.set_xticks(angles[:-1])
ax.set_xticklabels(construct_names, fontsize=10)
ax.set_yticks([20, 40, 60, 80, 100])
ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"], fontsize=8)
ax.set_ylim(0, 100)
ax.set_title("Percentage of High, Medium, and Low Category Respondents Across Open Data Constructs", size=14, weight="bold", pad=20)

# Add legend
ax.legend(loc="lower right", bbox_to_anchor=(1.2, -0.1), title="Category")

plt.tight_layout()
plt.show()

# Prepare data for heatmap
heatmap_data = np.array([low_percentages, medium_percentages, high_percentages])
categories = ["Low", "Medium", "High"]

# Create heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(
    heatmap_data,
    annot=True,
    fmt=".1f",
    cmap="coolwarm",
    xticklabels=construct_names,
    yticklabels=categories
)

# Add title and labels
plt.title("Heatmap of Percentage Respondent Categories Across Constructs")
plt.xlabel("Constructs")
plt.ylabel("Categories")

plt.tight_layout()
plt.show()

from scipy.stats import f_oneway, kruskal

# Prepare results
anova_results = {}
kruskal_results = {}

for construct_name, columns in constructs.items():
    # Calculate total scores for the construct
    scores = np.sum(columns, axis=1)
    
    # Classify respondents into Low, Medium, High
    low_scores = [score for score in scores if classify(score, construct_name) == "Low"]
    medium_scores = [score for score in scores if classify(score, construct_name) == "Medium"]
    high_scores = [score for score in scores if classify(score, construct_name) == "High"]
    
    # Perform ANOVA
    anova_stat, anova_p = f_oneway(low_scores, medium_scores, high_scores)
    anova_results[construct_name] = {"F-statistic": anova_stat, "p-value": anova_p}
    
    # Perform Kruskal-Wallis Test (non-parametric alternative)
    kruskal_stat, kruskal_p = kruskal(low_scores, medium_scores, high_scores)
    kruskal_results[construct_name] = {"H-statistic": kruskal_stat, "p-value": kruskal_p}

# Display Results
print("ANOVA Results:")
for construct, result in anova_results.items():
    print(f"{construct}: F = {result['F-statistic']:.2f}, p = {result['p-value']:.4f}")

print("\nKruskal-Wallis Test Results:")
for construct, result in kruskal_results.items():
    print(f"{construct}: H = {result['H-statistic']:.2f}, p = {result['p-value']:.4f}")
