import pdb
from dotenv import load_dotenv
load_dotenv()

from alphacopilot.api.calendars import advanceDateByCalendar
from alphacopilot.api.data import RetrievalAPI


start_date = '2025-01-01'
end_date = '2025-05-08'
start_time = advanceDateByCalendar('china.sse', start_date,
                                       '-{0}b'.format(1)).strftime('%Y-%m-%d')

end_time = advanceDateByCalendar('china.sse', end_date,
                                       '{0}b'.format(1)).strftime('%Y-%m-%d')


pdb.set_trace()
data = RetrievalAPI.get_main_price(begin_date=start_time,
                                       end_date=end_time,
                                       codes=['RB'],
                                       method='pcr',
                                       format_data=0)