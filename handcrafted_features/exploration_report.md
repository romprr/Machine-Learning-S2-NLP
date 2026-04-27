# Handcrafted Features Exploration Report

This report documents the analysis and evaluation of manually crafted regex and keyword rules applied to the dataset.

## Criteria for Acceptance (The 'Kept' Features)
- **Coverage:** Must match at least **200** posts (representing roughly 10% of the 2,000 target samples).
- **Precision (Accuracy):** Must be **>= 50%** (indicates that when the phrase appears, it's highly predictive of the intended tag).

## Summary
- Total Kept Features: **25**
- Total Discarded Features: **79**

## Details of Accepted Features
- `\bstd::` (Target: c++) [Coverage: 423 | Precision: 0.98]
- `\bcout\b` (Target: c++) [Coverage: 756 | Precision: 0.99]
- `\bvector\b` (Target: c++) [Coverage: 228 | Precision: 0.83]
- `\bString\[\]\s+args\b` (Target: java) [Coverage: 403 | Precision: 0.72]
- `\bSystem\.out\.println\b` (Target: java) [Coverage: 708 | Precision: 0.92]
- `\bdocument\.getElementById\b` (Target: javascript) [Coverage: 366 | Precision: 0.72]
- `\$_\w+` (Target: php) [Coverage: 553 | Precision: 0.85]
- `\becho\b` (Target: php) [Coverage: 773 | Precision: 0.81]
- `\bGROUP BY\b` (Target: sql) [Coverage: 320 | Precision: 0.51]
- `\bConsole\.WriteLine\b` (Target: c#) [Coverage: 231 | Precision: 0.84]
- `ruby\s+on\s+rails` (Target: ruby-on-rails) [Coverage: 237 | Precision: 0.98]
- `ActiveRecord` (Target: ruby-on-rails) [Coverage: 365 | Precision: 1.00]
- `has_many` (Target: ruby-on-rails) [Coverage: 252 | Precision: 1.00]
- `@interface` (Target: objective-c) [Coverage: 297 | Precision: 0.75]
- `@implementation` (Target: objective-c) [Coverage: 229 | Precision: 0.74]
- `NSString\s*\*` (Target: objective-c) [Coverage: 429 | Precision: 0.54]
- `ng-app` (Target: angularjs) [Coverage: 214 | Precision: 1.00]
- `ng-model` (Target: angularjs) [Coverage: 435 | Precision: 0.99]
- `ng-controller` (Target: angularjs) [Coverage: 276 | Precision: 1.00]
- `\$scope` (Target: angularjs) [Coverage: 882 | Precision: 0.99]
- `\$http` (Target: angularjs) [Coverage: 349 | Precision: 0.99]
- `angular\.module` (Target: angularjs) [Coverage: 421 | Precision: 1.00]
- `printf\s*\(` (Target: c) [Coverage: 1165 | Precision: 0.88]
- `\$\(\s*document\s*\)\.\s*ready` (Target: jquery) [Coverage: 312 | Precision: 0.79]
- `background-color:` (Target: css) [Coverage: 302 | Precision: 0.59]

---
## Feature Evaluation Log

### 1. Basic Dataset Statistics
- **Total Rows:** 40000
- **Columns:** post, tags

### 2. Standard Regex & Keyword Patterns Evaluation

#### Target tag: `python`
- **\bdef\b** (Target: `python`) - Precision: **0.48**, Coverage: **637** -> DISCARDED
- **\bimport\b** (Target: `python`) - Precision: **0.36**, Coverage: **516** -> DISCARDED
- **\bprint\b** (Target: `python`) - Precision: **0.42**, Coverage: **862** -> DISCARDED
- **\bself\b** (Target: `python`) - Precision: **0.12**, Coverage: **233** -> DISCARDED
- **\bdict\b** (Target: `python`) - Precision: **0.54**, Coverage: **66** -> DISCARDED
- **\blist\b** (Target: `python`) - Precision: **0.15**, Coverage: **488** -> DISCARDED
- **\bpandas\b** (Target: `python`) - Precision: **0.95**, Coverage: **18** -> DISCARDED
- **\bmatplotlib\b** (Target: `python`) - Precision: **1.00**, Coverage: **9** -> DISCARDED

#### Target tag: `c++`
- **#include\b** (Target: `c++`) - Precision: **0.46**, Coverage: **669** -> DISCARDED
- **\bstd::** (Target: `c++`) - Precision: **0.98**, Coverage: **423** -> KEPT
- **\bcout\b** (Target: `c++`) - Precision: **0.99**, Coverage: **756** -> KEPT
- **\bvector\b** (Target: `c++`) - Precision: **0.83**, Coverage: **228** -> KEPT

#### Target tag: `java`
- **\bpublic\s+class\b** (Target: `java`) - Precision: **0.49**, Coverage: **621** -> DISCARDED
- **\bString\[\]\s+args\b** (Target: `java`) - Precision: **0.72**, Coverage: **403** -> KEPT
- **\bSystem\.out\.println\b** (Target: `java`) - Precision: **0.92**, Coverage: **708** -> KEPT
- **\bArrayList\b** (Target: `java`) - Precision: **0.53**, Coverage: **157** -> DISCARDED
- **\bprintStackTrace\b** (Target: `java`) - Precision: **0.39**, Coverage: **81** -> DISCARDED

#### Target tag: `javascript`
- **\bvar\b** (Target: `javascript`) - Precision: **0.34**, Coverage: **1134** -> DISCARDED
- **\bfunction\s*\(** (Target: `javascript`) - Precision: **0.19**, Coverage: **572** -> DISCARDED
- **\bconsole\.log\b** (Target: `javascript`) - Precision: **0.43**, Coverage: **293** -> DISCARDED
- **\bdocument\.getElementById\b** (Target: `javascript`) - Precision: **0.72**, Coverage: **366** -> KEPT

#### Target tag: `php`
- **\$_\w+** (Target: `php`) - Precision: **0.85**, Coverage: **553** -> KEPT
- **\becho\b** (Target: `php`) - Precision: **0.81**, Coverage: **773** -> KEPT
- **\bpreg_match\b** (Target: `php`) - Precision: **0.95**, Coverage: **38** -> DISCARDED
- **\bmysqli\b** (Target: `php`) - Precision: **0.71**, Coverage: **20** -> DISCARDED

#### Target tag: `html`
- **<\s*div\b** (Target: `html`) - Precision: **0.23**, Coverage: **100** -> DISCARDED
- **<\s*span\b** (Target: `html`) - Precision: **0.54**, Coverage: **7** -> DISCARDED
- **<\s*a\b** (Target: `html`) - Precision: **0.08**, Coverage: **434** -> DISCARDED
- **<\s*html\b** (Target: `html`) - Precision: **1.00**, Coverage: **3** -> DISCARDED

#### Target tag: `sql`
- **\bSELECT\b** (Target: `sql`) - Precision: **0.27**, Coverage: **1153** -> DISCARDED
- **\bFROM\b** (Target: `sql`) - Precision: **0.10**, Coverage: **1313** -> DISCARDED
- **\bWHERE\b** (Target: `sql`) - Precision: **0.16**, Coverage: **972** -> DISCARDED
- **\bJOIN\b** (Target: `sql`) - Precision: **0.38**, Coverage: **459** -> DISCARDED
- **\bGROUP BY\b** (Target: `sql`) - Precision: **0.51**, Coverage: **320** -> KEPT

#### Target tag: `c#`
- **\bConsole\.WriteLine\b** (Target: `c#`) - Precision: **0.84**, Coverage: **231** -> KEPT
- **\busing\s+System** (Target: `c#`) - Precision: **0.63**, Coverage: **111** -> DISCARDED
- **\bnamespace\b** (Target: `c#`) - Precision: **0.18**, Coverage: **155** -> DISCARDED
- **\bIList\b** (Target: `c#`) - Precision: **0.43**, Coverage: **10** -> DISCARDED

### 3. Domain & URL Mentions Analysis

#### Target tag: `php`
- **php.net** (Target: `php`) - Precision: **0.94**, Coverage: **31** -> DISCARDED

#### Target tag: `java`
- **docs.oracle.com** (Target: `java`) - Precision: **0.86**, Coverage: **12** -> DISCARDED
- **spring.io** (Target: `java`) - Precision: **0.00**, Coverage: **0** -> DISCARDED

#### Target tag: `c#`
- **docs.microsoft.com** (Target: `c#`) - Precision: **0.00**, Coverage: **0** -> DISCARDED

#### Target tag: `javascript`
- **developer.mozilla.org** (Target: `javascript`) - Precision: **0.53**, Coverage: **16** -> DISCARDED

### 4. Robust API & Multi-Word phrase Evaluation

#### Target tag: `ruby-on-rails`
- **def\s+index** (Target: `ruby-on-rails`) - Precision: **0.98**, Coverage: **121** -> DISCARDED
- **class\s+\w+\s*<\s*ApplicationController** (Target: `ruby-on-rails`) - Precision: **0.00**, Coverage: **0** -> DISCARDED
- **ruby\s+on\s+rails** (Target: `ruby-on-rails`) - Precision: **0.98**, Coverage: **237** -> KEPT
- **rails\s+generate** (Target: `ruby-on-rails`) - Precision: **1.00**, Coverage: **21** -> DISCARDED
- **gem\s+[\'"]\w+[\'"]** (Target: `ruby-on-rails`) - Precision: **0.00**, Coverage: **0** -> DISCARDED
- **ActiveRecord** (Target: `ruby-on-rails`) - Precision: **1.00**, Coverage: **365** -> KEPT
- **has_many** (Target: `ruby-on-rails`) - Precision: **1.00**, Coverage: **252** -> KEPT

#### Target tag: `ios`
- **import\s+UIKit** (Target: `ios`) - Precision: **0.91**, Coverage: **10** -> DISCARDED
- **UIViewController** (Target: `ios`) - Precision: **0.48**, Coverage: **109** -> DISCARDED
- **viewDidLoad** (Target: `ios`) - Precision: **0.43**, Coverage: **115** -> DISCARDED
- **override\s+func** (Target: `ios`) - Precision: **0.88**, Coverage: **21** -> DISCARDED

#### Target tag: `objective-c`
- **#import\s+<** (Target: `objective-c`) - Precision: **0.00**, Coverage: **0** -> DISCARDED
- **@interface** (Target: `objective-c`) - Precision: **0.75**, Coverage: **297** -> KEPT
- **@implementation** (Target: `objective-c`) - Precision: **0.74**, Coverage: **229** -> KEPT
- **#import\s+[\'"]\w+** (Target: `objective-c`) - Precision: **0.00**, Coverage: **0** -> DISCARDED
- **\[self\s+\w+\]** (Target: `objective-c`) - Precision: **0.49**, Coverage: **116** -> DISCARDED
- **NSString\s*\*** (Target: `objective-c`) - Precision: **0.54**, Coverage: **429** -> KEPT
- **NSArray\s*\*** (Target: `objective-c`) - Precision: **0.46**, Coverage: **120** -> DISCARDED
- **NSDictionary\s*\*** (Target: `objective-c`) - Precision: **0.35**, Coverage: **65** -> DISCARDED

#### Target tag: `angularjs`
- **ng-app** (Target: `angularjs`) - Precision: **1.00**, Coverage: **214** -> KEPT
- **ng-model** (Target: `angularjs`) - Precision: **0.99**, Coverage: **435** -> KEPT
- **ng-controller** (Target: `angularjs`) - Precision: **1.00**, Coverage: **276** -> KEPT
- **\$scope** (Target: `angularjs`) - Precision: **0.99**, Coverage: **882** -> KEPT
- **\$http** (Target: `angularjs`) - Precision: **0.99**, Coverage: **349** -> KEPT
- **angular\.module** (Target: `angularjs`) - Precision: **1.00**, Coverage: **421** -> KEPT

#### Target tag: `iphone`
- **NSObject** (Target: `iphone`) - Precision: **0.08**, Coverage: **22** -> DISCARDED
- **CGRectMake** (Target: `iphone`) - Precision: **0.38**, Coverage: **87** -> DISCARDED
- **IBOutlet** (Target: `iphone`) - Precision: **0.23**, Coverage: **29** -> DISCARDED
- **IBAction** (Target: `iphone`) - Precision: **0.22**, Coverage: **55** -> DISCARDED

#### Target tag: `c`
- **#include\s+<stdio\.h>** (Target: `c`) - Precision: **1.00**, Coverage: **1** -> DISCARDED
- **#include\s+<stdlib\.h>** (Target: `c`) - Precision: **0.00**, Coverage: **0** -> DISCARDED
- **int\s+main\s*\(\s*\)** (Target: `c`) - Precision: **0.46**, Coverage: **482** -> DISCARDED
- **printf\s*\(** (Target: `c`) - Precision: **0.88**, Coverage: **1165** -> KEPT

#### Target tag: `mysql`
- **INSERT\s+INTO\s+\w+** (Target: `mysql`) - Precision: **0.37**, Coverage: **128** -> DISCARDED
- **AUTO_INCREMENT** (Target: `mysql`) - Precision: **0.90**, Coverage: **121** -> DISCARDED
- **mysql_query** (Target: `mysql`) - Precision: **0.14**, Coverage: **26** -> DISCARDED
- **phpMyAdmin** (Target: `mysql`) - Precision: **0.76**, Coverage: **39** -> DISCARDED

#### Target tag: `jquery`
- **\$\(\s*[\'"]#\w+** (Target: `jquery`) - Precision: **0.00**, Coverage: **0** -> DISCARDED
- **\$\(\s*function\s*\(** (Target: `jquery`) - Precision: **0.73**, Coverage: **96** -> DISCARDED
- **\$\.\s*ajax** (Target: `jquery`) - Precision: **0.70**, Coverage: **85** -> DISCARDED
- **\$\(\s*document\s*\)\.\s*ready** (Target: `jquery`) - Precision: **0.79**, Coverage: **312** -> KEPT

#### Target tag: `css`
- **\{\s*margin:** (Target: `css`) - Precision: **0.75**, Coverage: **128** -> DISCARDED
- **\{\s*padding:** (Target: `css`) - Precision: **0.77**, Coverage: **73** -> DISCARDED
- **background-color:** (Target: `css`) - Precision: **0.59**, Coverage: **302** -> KEPT
- **\.css\s*\(** (Target: `css`) - Precision: **0.04**, Coverage: **9** -> DISCARDED

#### Target tag: `sql`
- **SQL\s+Server** (Target: `sql`) - Precision: **0.28**, Coverage: **77** -> DISCARDED
- **FOREIGN\s+KEY** (Target: `sql`) - Precision: **0.37**, Coverage: **63** -> DISCARDED
- **PRIMARY\s+KEY** (Target: `sql`) - Precision: **0.31**, Coverage: **101** -> DISCARDED
- **VARCHAR\(** (Target: `sql`) - Precision: **0.34**, Coverage: **97** -> DISCARDED

#### Target tag: `c#`
- **public\s+class** (Target: `c#`) - Precision: **0.15**, Coverage: **194** -> DISCARDED
- **using\s+System** (Target: `c#`) - Precision: **0.62**, Coverage: **111** -> DISCARDED
- **static\s+void\s+Main** (Target: `c#`) - Precision: **0.21**, Coverage: **156** -> DISCARDED
- **Page_Load** (Target: `c#`) - Precision: **0.09**, Coverage: **14** -> DISCARDED
- **ToString\(\)** (Target: `c#`) - Precision: **0.30**, Coverage: **233** -> DISCARDED
- **Convert\.ToInt32** (Target: `c#`) - Precision: **0.55**, Coverage: **53** -> DISCARDED

#### Target tag: `asp.net`
- **asp:Button** (Target: `asp.net`) - Precision: **0.89**, Coverage: **74** -> DISCARDED
- **asp:Label** (Target: `asp.net`) - Precision: **0.91**, Coverage: **75** -> DISCARDED
- **Page_Load** (Target: `asp.net`) - Precision: **0.81**, Coverage: **129** -> DISCARDED
- **runat="server"** (Target: `asp.net`) - Precision: **0.00**, Coverage: **0** -> DISCARDED
- **System\.Web** (Target: `asp.net`) - Precision: **0.78**, Coverage: **166** -> DISCARDED
- **asp:GridView** (Target: `asp.net`) - Precision: **0.96**, Coverage: **52** -> DISCARDED
