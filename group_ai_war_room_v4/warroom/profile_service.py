from .data_service import profile_detail,metric_compare
from .utils import mean

PROFILE_COLORS={'basic':'#34e8ff','operation':'#5cf2b5','technology':'#a884ff','supervision':'#ffb45b'}


def profile_payload(company_id,profile):
    data=profile_detail(company_id,profile)
    for m in data['metrics']:
        comp=m['compare']['items'];m['honeycomb']=[{'company_id':x['company_id'],'company_name':x['company_name'],'value':x['value'],'rank':x['rank'],'percentile':x['percentile']} for x in comp]
    data['color']=PROFILE_COLORS[profile]
    return data
