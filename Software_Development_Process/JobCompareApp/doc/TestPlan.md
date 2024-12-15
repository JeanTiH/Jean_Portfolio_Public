# Test Plan
**Author**: _Team 173_

**Version D3**
- Finalized the Testing Strategy, including updateing team members' role for testing,  enriching the adequacy criterion, and updating the techonologies used for testing.
- Updated the Test Cases section with final results, including both JUnit tests and integration+system tests.

## 1 Testing Strategy

### 1.1 Overall strategy

We will perform tests on multiple granularity levels, including unit testing, integration testing, system testing, and regression testing:
- Unit Testing: we will test individual methods within each class ('JobManager', 'CurrentJobLayoutUI', etc.) separately to make sure they work as expected. The team will use JUnit to do unit testing.
- Integration Testing: we will conduct integration tests to ensure different classes and modules interact correctly, specifically the interactions between the UI classes and the 'JobManager' (MainMenu) and the functionalities of the 'JobComparator' class.
- System Testing: the team will test if the application as a whole works as expected. We will test to make sure every requirement listed can be satisfied, including both the user interface and functions.
- Regression Testing: we will perform regression tests every time we make changes to verify that existing functionality remains unaffected.

All team members will participate in the testing in a collaborative way：Juejing Han 
- JUnit Tests and Regression Tests: Juejing Han, Erik Lee, Michael Lukacsko, Yelin Qin
- Integration Tests and System Tests: Yelin Qin
- Test Documentation: Yelin Qin

### 1.2 Test Selection

The team plans to use a combination of white-box and black-box techniques for this project, specifically:
- We will use white-box techniques during unit testing to review the logic of the code and make sure it has been correctly implemented.
- We will use black-box techniques for integration testing and system testing to test if we get the same expected output as described in the requirements.

### 1.3 Adequacy Criterion

The team plans to assess the quality of test cases through both functional and structural coverage:
- We will assess the quality of unit testing by structural coverage metrics, such as branch coverage and statement coverage, to make sure each branch and statement in the code is executed. The goal is to ensure every branch and statement that are non-navigational is covered by unit testing.
- We will assess the quality of integration testing and system testing by functional coverage metrics to make sure all features and requirements are tested, such as all scenarios of entering job details, entering or editing current jobs, comparing jobs, and changing comparison settings are appropriately tested. 

### 1.4 Bug Tracking

Bugs and enhancement requests will be tracked using Github's Issues tool, which is a useful management tool that allows the whole team to track bugs. We will submit new Issues reports with how we observe the bug, the expected vs actual outcomes, and the priority of the issue. We will also prioritize across the bugs and update the status timely.

### 1.5 Technology
- We will use JUnit for unit testing.
- We will use manual testing techniques for integration testing and system testing.
- We will use both JUnit and manual testing for regression testing.

## 2 Test Cases
| No. | Purpose | Steps | Expected Results | Actual Result | Pass/Fail | Notes|
|---|---|---|---|---|---|---|
|1| main menu unit tests | JUnit tests for the status of Compare Job button| only enabled when at least 2 jobs (including current job) saved | JUnit test MainActivityTest passed | Passed | |
|2| data manager unit tests | JUnit tests for DataManager methods | the set() methods save changes and the get() methods retrieve data correctly for current job and job offers | JUnit test DataManagerTest passed | Passed | |
|3| unit tests for editing current job | JUnit tests for CurrentJobActivity | be able to edit the 8 job parameters | JUnit test CurrentJobActivityTest passed | Passed | |
|4| unit tests for adding job offers | JUnit tests for JobOfferActivity | be able to edit the 8 job parameters and add new offer to the list of offers | JUnit test JobOfferActivityTest passed | Passed | |
|5| unit tests for adjusting comparison setting | JUnit tests for WeightSettingsActivity | be able to edit the weights (0-9 integer) for the 5 factors | JUnit test WeightTest passed | Passed | |
|6| unit tests for calculating job score | JUnit tests for JobScorer| given a set of Job parameters and CompareOfferSettings parameters, test whether the score is correctly calculated as weighted (AYS + AYB + TDF + (LT * AYS / 260) - ((260 - 52 * RWT) * (AYS / 260) / 8))| JUnit test JobScorerTest Passed | Passed | |
|7| unit tests for ranking jobs | JUnit tests for CompareJobActivity | rank the offers (including the current job) from high to low using the JobScorer and allowing for selecting 2 jobs to compare | JUnit test CompareJobActivityTest passed | Passed | |
|8| unit tests for compare 2 jobs | JUnit tests for CompareTwoJobsActivity | show a table with the 8 Job parameters for the two jobs selected | JUnit test CompareTwoJobsActivityTest passed | Passed | |
|9| unit tests for compare the newly added job offer with current job | JUnit tests for CompareWithCurrentActivity | show a table with the 8 Job parameters for the newly added job offer and current job | JUnit test CompareWithCurrentActivityTest passed  | Passed | |
|10|test main menu UI|start the app| the main menu should be shown with the 4 options to edit or enter the current job, enter job offers, adjust comparison settings, and compare job offers (this should be disabled when no job offers entered) |as expected (manually tested) | Passed | |
|11| main menu navigation tests | launch the app and click each of the 4 buttons on MainMenu| each button should result in navigation to the desired UI | as expected (manually tested) | Passed | |
|12| tests for save, clear, cancel, and return to main menu | tests for the navigational methods for each UI class| save will update the relevant attributes, clear will clean the inputs, cancel will return to main menu without saving, return will return to main menu | as expected (manually tested) | Passed | |
|13|test current job interface |choose to enter current job details|show the options to enter or edit the 8 types of details |as expected (manually tested) | Passed | |
|14|test save current job|after landing on the current job UI, save the job details| job details should be saved and return to the main menu |as expected (manually tested) | Passed  | |
|15|test exit current job|after landing on the current job UI, cancel without saving| no change to the current job and return to the main menu |as expected (manually tested) | Passed | |
|16|test job offer interface |choose to enter job offers |show the options to enter the 8 types of details | as expected (manually tested) | Passed | |
|17|test save job offer |after landing on the job offer UI, enter all details and save| job details should be saved and show options to return to the main menu, enter another offer, or compare with current job if present |as expected (manually tested) | Passed | |
|18|test exit job offer|after landing on the job offer UI, cancel without saving| no change to the job offers and return to the main menu|as expected (manually tested) | Passed | |
|19|test comparison setting interface |choose to adjust comparison settings from main menu |show the options to assign weights (0-9) to the 5 factors| as expected (manually tested) | Passed | |
|20|test compare offer interface |choose to compare job offers from main menu |show the list of job offers (title+company) ranked from best to worse by the scores, with the current job indicated, and the option to select 2 jobs to compare|as expected (manually tested) | Passed | |
|21|test 2 jobs comparison | trigger 2 job comparison from the compare job offer interface or after entering a new job offer|show a table that compares the 8 attributes of the selected jobs| as expected (manually tested) | Passed | |
|22|test post comparison |after compare 2 jobs| show the option to perform another comparison or go back to the main menu|as expected (manually tested) | Passed | |