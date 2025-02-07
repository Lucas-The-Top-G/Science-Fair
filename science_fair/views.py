import django
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.views import View
from .utils import get_arr_to_string
from play import get_response

@method_decorator(csrf_exempt, name='dispatch')
def stream_response(request):
    if request.method == 'GET':
        headers = request.headers

        exterior_data = get_arr_to_string(headers.get('Exterior', ''))
        parking_data = get_arr_to_string(headers.get('Parking', ''))
        heating_data = get_arr_to_string(headers.get('Heating', ''))

        try:
            Price = int(headers.get('Price', 0))
            SQFT = int(headers.get('SQFT', 0))
            Acres = int(headers.get('Acres', 0))
            Beds = int(headers.get('Beds', 0))
            Baths = int(headers.get('Baths', 0))
        except ValueError:
            return JsonResponse({'error': 'Invalid number format in headers'}, status=400)

        Sub = headers.get('Sub', '')
        Description = headers.get('Description', '')

        res = get_response(Price, SQFT, Beds, Baths, Acres, exterior_data, heating_data, Sub, parking_data, Description)

        return JsonResponse({'response': res})

    return JsonResponse({'error': 'Invalid request method'}, status=405)
