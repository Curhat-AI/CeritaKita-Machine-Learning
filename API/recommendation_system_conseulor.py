import firebase_admin
from firebase_admin import firestore
from google.oauth2 import service_account
import pandas as pd

cred = firebase_admin.credentials.Certificate("serviceaccountKey-firebase-adminsdk.json")
firebase_admin.initialize_app(cred)
db = firestore.client()
credentials = service_account.Credentials.from_service_account_file("serviceaccountKey-firebase-adminsdk.json")

def recommendation_result(input):
    users = list(db.collection('users').stream())
    users_id = list(map(lambda x: x.id, users))
    users_dict = list(map(lambda x: x.to_dict(), users))
    df_user = pd.DataFrame(users_dict)
    df_user['user_id'] = users_id
    data = df_user['counselorDetails']
    data = data.where(data.notna(), lambda x: [{}])
    df_counselour_details = pd.DataFrame.from_dict(data.to_list())
    df_counselour_details['counselorId'] = df_user['user_id']
    df_user_detail = df_user["details"]
    df_user_detail = df_user_detail.where(df_user_detail.notna(), lambda x: [{}])
    df_user_details = pd.DataFrame.from_dict(df_user_detail.to_list())
    df_user_details['counselorId'] = df_user['user_id']
    counseling_detail = list(db.collection('counselingSessions').stream())
    counseling_detail_dict = list(map(lambda x: x.to_dict(), counseling_detail))
    df_counseling_detail = pd.DataFrame(counseling_detail_dict)
    df_counseling_detail = df_counseling_detail["counselingDetails"]
    df_counseling_detail = df_counseling_detail.where(df_counseling_detail.notna(), lambda x: [{}])
    df_counseling_details = pd.DataFrame.from_dict(df_counseling_detail.to_list())
    data = pd.merge(df_counselour_details, df_counseling_details, on = "counselorId", how='left')
    data = pd.merge(data, df_user_details, on = "counselorId", how='left')
    gender = input.gender
    status_konselor = input.counselourType
    tanggal_bawah = pd.to_datetime(input.dateDown, format='%Y-%m-%d')
    tanggal_atas = pd.to_datetime(input.dateUp, format='%Y-%m-%d')
    waktu_bawah = input.timeDown
    waktu_atas = input.timeUp
    recommendation_counselor = data.copy() 

    recommendation_counselor = recommendation_counselor[(recommendation_counselor['gender'] == gender) &
                (recommendation_counselor['counselorType'] == status_konselor) &
                (recommendation_counselor['startTime'].dt.date >= tanggal_bawah.date()) &
                (recommendation_counselor['endTime'].dt.date <= tanggal_atas.date()) &
                (int(str(recommendation_counselor['endTime'][0].time())[0:2]) >= waktu_bawah) &
                (int(str(recommendation_counselor['endTime'][0].time())[0:2]) <= waktu_atas+0.5)]

    recommendation_counselor = recommendation_counselor.sort_values('rating', ascending=False)

    result = recommendation_counselor['counselorId'].unique().tolist()
    return result