from django.shortcuts 				import render,redirect

def homeView(request):
	return render(request,'pages/home.html')

	