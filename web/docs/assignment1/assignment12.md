# Task N2. Cross-Entropy and Language Modeling
## English Language
Number of unique words in texten1 9607
### English datasets
Data: 221098
Train data: 161098
Test data: 20000
Heldout data: 40000
### Compute parameters from the heldout data 
Smoothing params for heldout  0.001534102378745178, 0.17295027933403023, 0.6253429555482346, 0.2001726627389901
Cross entropy with lambdas 7.020780645922868
### Experiments  
Adding 10%, 20%, 30%, ..., 90%, 95% and 99% of the difference between the trigram smoothing parameter and 1.0 to its value *discounting* other parameters proportionally

|         Percent          |         Entropy          |
| ------------------------ | ------------------------ |
| 10                       | 7.029464955658704        |
| 20                       | 7.055005563786017        |
| 30                       | 7.096182412069876        |
| 40                       | 7.153900741196933        |
| 50                       | 7.231063676376062        |
| 60                       | 7.333447220810907        |
| 70                       | 7.472329835578622        |
| 80                       | 7.672588785291019        |
| 90                       | 8.008293544669762        |
| 95                       | 8.319400505488423        |
| 99                       | 8.90003989542219         |

----------------------------------------------------------------------------------------
Setting trigram smoothng parameter to 90%, 80%, 70%, ... 10%, 0% of its value, *boosting* other parameters proportionally

|         Percent          |         Entropy          |
| ------------------------ | ------------------------ |
| 0                        | 7.44024621043973         |
| 10                       | 7.233885093742739        |
| 20                       | 7.090030291796269        |
| 30                       | 6.968128403985225        |
| 40                       | 6.8605930495629845       |
| 50                       | 6.763758992427127        |
| 60                       | 6.675402858174792        |
| 70                       | 6.594015599498247        |
| 80                       | 6.5185020188407785       |
| 90                       | 6.448031028725916        |

