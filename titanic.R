# Load Library
library(tidymodels)
library(tidyverse)

# Import dataset
titanic_test <- read_csv("test.csv")
titanic_train <- read_csv("train.csv")

titanic_train %>% summarise_all(~ sum(is.na(.)))
titanic_test %>% summarise_all(~ sum(is.na(.)))

titanic_full <- bind_rows(titanic_train, titanic_test)

titanic_full %>% summarise_all(~ sum(is.na(.)))

glimpse(titanic_full)
str(titanic_full)

# Data transformation
titanic_full <- titanic_full %>%
  mutate(FamilySize = SibSp + Parch + 1) %>%
  select(-SibSp, -Parch)

titanic_full$title <- titanic_full$Name %>%
  str_extract("([A-z]+)\\.") %>%
  str_sub(end = -2)
titanic_full %>%
  group_by(title) %>%
  count() |> 
  arrange(desc(n)) |> 
  ungroup() |> 
  mutate(prop = n/sum(n))

titanic_full <- titanic_full %>% select(-Cabin, -Ticket, -Name)

titanic_full <- titanic_full %>% mutate(across(c(Survived, Pclass, Sex, Embarked, title), as.factor))

# input missing data
impute_data <- recipe(Survived ~ ., data = titanic_full) %>%
  step_impute_mode(Embarked) %>%
  step_impute_linear(Fare) %>%
  step_impute_knn(Age) %>% 
  prep()

imputed <- bake(impute_data, new_data = titanic_full)

imputed %>% summarise_all(~ sum(is.na(.)))

imputed <- imputed %>% mutate(Age = case_when(
  Age < 8 ~ "anak-anak",
  Age >= 9 & Age < 19 ~ "remaja",
  Age >= 19 & Age <= 60 ~ "dewasa",
  TRUE ~ "lansia"
))

imputed$Age <- as.factor(imputed$Age)

glimpse(imputed)

# build a model
titanic_train <- imputed %>% slice(1:891)
titanic_test <- imputed %>% slice(892:1309)

imputed |> view()

set.seed(777)
split <- initial_split(titanic_train, prop = 0.9, strata = Survived)
train <- training(split)
test <- testing(split)

titanic_recipe <- recipe(data = train, formula = Survived ~ .) %>%
  update_role(PassengerId, new_role = 'id') |> 
  step_normalize(all_numeric_predictors()) |> 
  step_other(title, threshold = 0.02) |> 
  step_dummy(all_nominal_predictors()) |> 
  prep()

bake(titanic_recipe, new_data = NULL) 

set.seed(345)
folds <- vfold_cv(train, v = 10, repeats = 1, strata = Survived)

rf_spec <- rand_forest(mtry = tune(), min_n = tune(), trees = 1000) %>%
  set_engine("ranger") %>%
  set_mode("classification")

titanic_workflow <- workflow() %>%
  add_recipe(titanic_recipe) %>%
  add_model(rf_spec)

rf_tune <- tune_grid(titanic_workflow,
  resamples = folds,
  grid = 10
)

# evaluate model
collect_metrics(rf_tune)

show_best(rf_tune, metric = 'accuracy')

final_rf <- titanic_workflow %>%
  finalize_workflow(select_best(rf_tune, metric = 'accuracy'))

final_fit <- final_rf |> 
  last_fit(split)

collect_metrics(final_fit)

final_model <-  extract_workflow(final_fit)

pred <- final_model |> predict(titanic_test)
pred

prediksi <- titanic_test |> 
  select(PassengerId) |> 
  bind_cols(pred) |> 
  rename(Survived = .pred_class)

write_csv(prediksi, file = "C:/Users/rokub/Desktop/predict.csv")
