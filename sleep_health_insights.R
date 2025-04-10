# =====================================================
# Project: Sleep Health Insights
# Author: Sheila Houmey (@softlyshebuilds)
# Description: Exploratory data analysis and BMI category prediction
# Date: November 2024 – February 2025
# Dataset: Sleep Health and Lifestyle Dataset (Kaggle)
# =====================================================

# Load Libraries
library(caret)
library(rpart)
library(ggplot2)

# Load Dataset
df <- read.csv("data/sleep_fitness_internal_dataset.csv")

# BMI Category Cleaning
df$`BMI.Category` <- gsub("^Normal$", "Normal Weight", df$`BMI.Category`)
df$`BMI.Category` <- as.factor(df$`BMI.Category`)
levels(df$`BMI.Category`) <- c("Normal Weight", "Overweight", "Obese")

# BMI Category Distribution Plot
ggplot(df, aes(x = `BMI.Category`, fill = `BMI.Category`)) + 
  geom_bar() +
  labs(title = "Distribution of BMI Categories", 
       x = "BMI Category", 
       y = "Count") +
  theme_minimal()

# Outlier Detection: Daily Steps
daily_steps <- df$Daily.Steps
Q1 <- quantile(daily_steps, 0.25)
Q3 <- quantile(daily_steps, 0.75)
IQR <- Q3 - Q1
lower_bound <- Q1 - 1.5 * IQR
upper_bound <- Q3 + 1.5 * IQR
outliers <- daily_steps[daily_steps < lower_bound | daily_steps > upper_bound]

cat("Q1: ", Q1, " | Q3: ", Q3, " | IQR: ", IQR, "\n")
cat("Outliers in Daily Steps: ", outliers, "\n")

# Remove outliers for training
df_cleaned <- df[daily_steps >= lower_bound & daily_steps <= upper_bound, ]

# = Data Split =
set.seed(42)
trainIndex <- createDataPartition(df_cleaned$`BMI.Category`, p = 0.8, list = FALSE)
trainData <- df_cleaned[trainIndex, ]
testData <- df_cleaned[-trainIndex, ]

# Outlier Detection: Sleep Duration
sleep_duration <- df$Sleep.Duration
Q1_sleep <- quantile(sleep_duration, 0.25)
Q2_sleep <- quantile(sleep_duration, 0.50)
Q3_sleep <- quantile(sleep_duration, 0.75)
IQR_sleep <- Q3_sleep - Q1_sleep
cat("Sleep Duration → Q2: ", Q2_sleep, ", IQR: ", IQR_sleep, "\n")

# Outlier Detection: Stress Level
stress_level <- df$Stress.Level
Q1_stress <- quantile(stress_level, 0.25)
Q2_stress <- quantile(stress_level, 0.50)
Q3_stress <- quantile(stress_level, 0.75)
IQR_stress <- Q3_stress - Q1_stress
cat("Stress Level → Q2: ", Q2_stress, ", IQR: ", IQR_stress, "\n")

# Model Training
model <- rpart(`BMI.Category` ~ Daily.Steps + Sleep.Duration + Stress.Level,
               data = trainData, method = "class")

# Evaluation
predictions <- predict(model, testData, type = "class")
predictions <- factor(predictions, levels = levels(testData$`BMI.Category`))
conf_matrix <- confusionMatrix(predictions, testData$`BMI.Category`)
print(conf_matrix)

# Final Accuracy
accuracy <- conf_matrix$overall['Accuracy']
cat("\n✅ Pipeline executed successfully. Overall Accuracy: ",
    round(accuracy * 100, 2), "%\n")
