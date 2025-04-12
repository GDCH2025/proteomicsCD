library(readr)
library(dplyr)
library(survival)

CDallpredictor <- read_csv("/GDCH/proteinAI/data/revise/CDallpredictor.csv")
CDallpredictor_train<-filter(CDallpredictor,`Region_code`==0)

cumulative_cox <- data.frame()

for (i in 3:2739) {
  
  time<-CDallpredictor_train %>% mutate(time = as.numeric(
    difftime(endTime, beginTime, units = "days")) / 365.25)
  
  alldata<-filter(time,time>0)
  
  catvars<-c("sex","Ethnicity")
  alldata[catvars] <- lapply(alldata[catvars], factor)
  
  tryCatch({
    mycox <- coxph(Surv(time, CD== 1) ~ CDallpredictor_train[[i]]+age+sex+Ethnicity, data = alldata)
    
    test.ph <- cox.zph(mycox)
    test.phtable <- data.frame(test.ph$table) 
    test.phtable <- test.phtable[1,3]
    
    summary_df <- data.frame(summary(mycox)$coefficients)
    confint_df <- data.frame(exp(confint(mycox, level = 0.95)))
    summary_df$exp_coef <- exp(summary_df[,1])
    summary_df$exp_lower<- confint_df[,1]
    summary_df$exp_upper<- confint_df[,2]
    summary_df<-summary_df[1,5:8]
    colnames(summary_df)[1] = 'Pvalue'
    summary_df$allevent<-mycox$nevent
    summary_df$total<-mycox$n
    summary_df$ph_Test<-test.phtable
    summary_df$exposure<-names(CDallpredictor_train)[i]
    summary_df$outcome <- "CD"
    
    cumulative_cox <- rbind(cumulative_cox, summary_df)
  }, error = function(e) {
    cat("Error occurred:", conditionMessage(e), "\n")
  })
} 
cumulative_cox$Pvalue_bonferroni<-p.adjust(cumulative_cox$Pvalue, method = "bonferroni")
cumulative_cox$Pvalue_fdr<-p.adjust(cumulative_cox$Pvalue, method = "fdr")
write_csv(cumulative_cox, file = "/GDCH/proteinAI/result1/cox/CDmodel1cox.csv")
