from django.urls import path
from .views import stream_response

urlpatterns = [
    path('process/', stream_response, name='process'),
]