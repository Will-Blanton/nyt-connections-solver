# Connections group categories by color

## Dataset and method

This exploratory breakdown uses `data/connections.csv`, which contains 4,852 groups across 1,213 puzzle dates (four groups per date). The CSV has no color column. Colors here are assigned by each group's position in the date's four rows, using the order confirmed for this dataset:

1. Purple
2. Blue
3. Green
4. Yellow

The qualitative observations below come from a reproducible random sample of 80 groups from each position (seed `20261010`). The examples are sampled category labels and their intended relationships. These are tendencies, not exclusive rules, and category classes can overlap.

## Purple: wordplay and phrase construction

Purple groups most often make the player transform, complete, or reinterpret words. Common forms include:

- Completing a phrase: `double ___`, `___ call`, or words before “stool.”
- Sound relationships: homophones and rhymes.
- Letter changes: adding or removing letters, or changing a starting letter.
- Shared beginnings or endings hidden across otherwise unrelated words.
- A word or phrase with several distinct meanings, such as what “buff” or “diamond” might mean.

Examples from the sample include “philosopher homophones,” “currencies plus a letter,” “ending in NBA players,” and “words before ‘fingers.’”

**Useful heuristic:** ask what operation or phrase template connects the words.

## Blue: named sets and knowledge-based associations

Blue includes many pop-culture groups, along with other sets that depend on recognizing people, works, brands, or established domains. Sampled examples include:

- Movies and television: ’80s comedies, Stephen King film adaptations, Hitchcock movies, and PBS shows.
- People and characters: Tom Hanks roles, fictional spies, famous poets, and country-music legends.
- Brands and named products: Dodge models, shoe brands, and toy-car brands.
- Other factual sets: tennis scores, golf clubs, and Roman numerals.

Pop culture is a noticeable part of blue, but it is not the whole category. A better general clue is that blue often asks the solver to retrieve a recognized set from cultural or domain knowledge.

**Useful heuristic:** look for a set whose members are established names, roles, works, or terms.

## Green: familiar categories and indirect relationships

Green often groups recognizable objects, parts, properties, functions, or moderately indirect synonyms. Sampled examples include:

- Parts and components: parts of a bra and features of a car's center console.
- Types: kinds of socks, trees, card games, and shades of green.
- Shared properties or settings: things made of wax, things with wings, and things seen in a barn.
- Actions or related meanings: “emit light,” “collide with,” and “guide.”

**Useful heuristic:** identify a conventional category or relationship, allowing for a little more interpretation than a direct synonym set.

## Yellow: direct meanings and familiar taxonomies

Yellow most often uses straightforward synonyms, everyday categories, and common examples. Sampled examples include:

- Synonyms: “beginning,” “suppress,” “promote,” and “affluent.”
- Familiar types: spices, cheeses, boats, dog breeds, and pianos.
- Everyday sets: body parts, playing cards, precipitation, and things to grab before leaving home.

**Useful heuristic:** start with the plain meaning of each word or an obvious everyday class.

## Overall pattern

The sampled groups suggest a rough progression in the kind of connection to look for:

| Color | Common solving angle |
|---|---|
| Yellow | Direct meaning or familiar category |
| Green | Familiar category with a less direct relationship |
| Blue | Named set or cultural/domain knowledge |
| Purple | Word construction, sound, or phrase manipulation |

This is a solving heuristic, not a rule for every puzzle. A blue group can be an ordinary category, and a purple group can depend on cultural knowledge. Since the CSV does not explicitly record colors, conclusions also depend on the confirmed row-order mapping above.
