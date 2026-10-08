# Data

No data files are committed: the dataset was provided by the course and is not redistributed. Place the files in `data/raw/` (excluded by `.gitignore`).

| File | Content |
|------|---------|
| `train.csv` | 3,200 labelled recipes: target `y` (1 = American, 2 = Italian) + 40 ingredient features |
| `test.csv` | 1,734 unlabelled recipes: the same 40 features |

**Features** (TF-IDF scores, non-negative, mostly zero): salt, sugar, water, cereals, oil, flour, fruits, milk, seeds, onion, garlic, chocolate, yeast, egg, vinegar, tomato, cream, rice, corn, cheese, butter, red.meat, white.meat, potato, honey, paprika, mustard, turmeric, ginger, parsley, wine, chili, mushrooms, bread, pasta, fish, seafood, veggies, legumes, pepper.

x<sub>ij</sub> = TF<sub>ij</sub> × IDF<sub>j</sub>. TF<sub>ij</sub> is the frequency of ingredient j in recipe i, normalised by the total number of ingredient mentions in that recipe. IDF<sub>j</sub> = log(4934 / n<sub>j</sub>), where n<sub>j</sub> is the number of recipes (out of 4,934) containing ingredient j.

**Note:** CSV headers may contain a byte-order mark (`\ufeff`); the notebook strips it.
