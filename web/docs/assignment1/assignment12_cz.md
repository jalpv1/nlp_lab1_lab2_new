## Czech Language
Number of unique words in textcz1 42826
### Czech datasets
Data: 222412
Train data: 162412
Test data: 20000
Heldout data: 40000
### Compute parameters from the heldout data 
Smoothing params for heldout  0.008563364408127216, 0.44157675662605506, 0.42073418776130683, 0.12912569120451092
Cross entropy with lambdas 9.699194379883629
### Experiments  
Adding 10%, 20%, 30%, ..., 90%, 95% and 99% of the difference between the trigram smoothing parameter and 1.0 to its value *discounting* other parameters proportionally

|         Percent          |         Entropy          |
| ------------------------ | ------------------------ |
| 10                       | 9.707930582775074        |
| 20                       | 9.739683039742795        |
| 30                       | 9.78996867243673         |
| 40                       | 9.858748874867823        |
| 50                       | 9.948786323493968        |
| 60                       | 10.066226186265439       |
| 70                       | 10.223388630219487       |
| 80                       | 10.44780292022898        |
| 90                       | 10.822894319300016       |
| 95                       | 11.174287698067806       |
| 99                       | 11.863545154661594       |

----------------------------------------------------------------------------------------
Setting trigram smoothng parameter to 90%, 80%, 70%, ... 10%, 0% of its value, *boosting* other parameters proportionally

|         Percent          |         Entropy          |
| ------------------------ | ------------------------ |
| 0                        | 9.98150154943724         |
| 10                       | 9.805792636318552        |
| 20                       | 9.696149938898259        |
| 30                       | 9.603799196055927        |
| 40                       | 9.522641632387968        |
| 50                       | 9.449914692480217        |
| 60                       | 9.384002858587543        |
| 70                       | 9.323844296924152        |
| 80                       | 9.26869951242026         |
| 90                       | 9.218043162718361        |

