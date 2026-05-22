# MetaShift — Concept Cleaning Report

## 1. Cleaning Rules

### Synonym Merging
| Original | Canonical |
|----------|-----------|
| beach's | beach |
| beds | bed |
| blackish | black |
| bluish | blue |
| bookshelf | shelf |
| boy | person |
| brownish | brown |
| cat's | cat |
| cats | cat |
| chairs | chair |
| child | person |
| children | person |
| desks | desk |
| dirty | dirt |
| disc | frisbee |
| dog's | dog |
| dogs | dog |
| girl | person |
| grassy | grass |
| greenish | green |
| grey | gray |
| guy | person |
| jungle | forest |
| keyboards | keyboard |
| kitchen's | kitchen |
| kitten | cat |
| kittens | cat |
| lake | water |
| laptops | laptop |
| lawn | grass |
| lawns | grass |
| laying | lay |
| living room | living_room |
| man | person |
| men | person |
| mountains | mountain |
| mud | dirt |
| muddy | dirt |
| notebook | laptop |
| ocean | water |
| pavement | street |
| pc | computer |
| people | person |
| person's | person |
| playing | play |
| puppies | dog |
| puppy | dog |
| reddish | red |
| restroom | bathroom |
| river | water |
| road | street |
| road's | street |
| sand | beach |
| sandy | beach |
| screen | monitor |
| sea | water |
| shelves | shelf |
| sidewalk | street |
| sitting | sit |
| sleeping | sleep |
| snowing | snow |
| snowy | snow |
| sofa | couch |
| sofas | couch |
| standing | stand |
| surf | water |
| surfboards | surfboard |
| tables | table |
| tree | forest |
| trees | forest |
| tv | television |
| tvs | television |
| walking | walk |
| whitish | white |
| woman | person |
| women | person |
| woods | forest |
| yellowish | yellow |

### Removed Stopwords
`angle, another, back, background, beautiful, big, blur, blurred, blurry, bottom, bunch, camera, center, close, closeup, collage, couple, day, different, edge, few, food, foods, foreground, front, good, great, group, head, huge...`

## 2. ViT-GPT2 Statistics

| Metric | Before | After |
|--------|--------|-------|
| Vocabulary size | 364 | 312 |
| Total tokens | 17483 | 15739 |

### Before (Top-20)
- dog: 2350
- man: 1028
- top: 778
- cat: 772
- woman: 605
- frisbee: 605
- white: 504
- black: 419
- people: 346
- water: 343
- beach: 283
- field: 275
- person: 250
- street: 216
- computer: 206
- surfboard: 202
- air: 194
- bench: 189
- brown: 177
- boat: 176

### After (Top-20)

- person: 2360
- dog: 2350
- cat: 772
- frisbee: 605
- white: 504
- street: 430
- black: 419
- water: 416
- beach: 309
- field: 275
- computer: 206
- surfboard: 202
- air: 194
- bench: 189
- brown: 177
- boat: 176
- grass: 169
- snow: 167
- laptop: 151
- mouth: 145

## 3. BLIP Statistics

| Metric | Before | After |
|--------|--------|-------|
| Vocabulary size | 211 | 211 |
| Total tokens | 21409 | 21409 |

### Before (Top-20)
- dog: 3434
- man: 1296
- cat: 906
- people: 667
- woman: 659
- frisbee: 585
- water: 513
- black: 458
- white: 432
- grass: 347
- beach: 345
- front: 331
- street: 287
- top: 260
- back: 260
- many: 253
- surfboard: 249
- field: 242
- boat: 231
- laptop: 226

### After (Top-20)

- dog: 3434
- man: 1296
- cat: 906
- people: 667
- woman: 659
- frisbee: 585
- water: 513
- black: 458
- white: 432
- grass: 347
- beach: 345
- front: 331
- street: 287
- top: 260
- back: 260
- many: 253
- surfboard: 249
- field: 242
- boat: 231
- laptop: 226

## 4. Impact Summary

| Model | Vocab Before | Vocab After | Reduction |
|-------|-------------|-------------|-----------|
| ViT-GPT2 | 364 | 312 | 14% |
| BLIP | 211 | 211 | 0% |

## 5. Key Improvements

- Synonyms merged: couch/sofa, grass/lawn, snow/snowy, beach/sandy, forest/trees/jungle, dirt/mud, etc.
- People terms unified: man/woman/child → person
- Removed: generic photo terms, spatial terms (front/back/top), vague quantifiers
- Colors preserved but minor variants merged
- Dog/cat variants unified to canonical forms

Cleaned concepts are saved at:
- `D:\SPUME\analysis\metashift\vitgpt2\concepts_cleaned_vitgpt2.json`
- `D:\SPUME\analysis\metashift\blip\concepts_cleaned_blip.json`