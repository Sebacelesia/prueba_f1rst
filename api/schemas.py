from pydantic import BaseModel


class EmployeeInput(BaseModel):
    hrs: float
    absences: int
    JobInvolvement: float
    PerformanceRating: int
    EnvironmentSatisfaction: float
    JobSatisfaction: float
    WorkLifeBalance: float
    Age: float
    BusinessTravel: str
    Department: str
    DistanceFromHome: int
    Education: int
    EducationField: str
    Gender: str
    JobLevel: int
    JobRole: str
    MaritalStatus: str
    MonthlyIncome: float
    NumCompaniesWorked: float
    PercentSalaryHike: int
    StockOptionLevel: int
    TotalWorkingYears: float
    TrainingTimesLastYear: int
    YearsAtCompany: int
    YearsSinceLastPromotion: int
    YearsWithCurrManager: int


class PredictionOutput(BaseModel):
    prediction: int
    prediction_label: str
    probability: float
