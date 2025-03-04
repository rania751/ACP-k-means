from django.shortcuts 				import render,redirect
from django.contrib.auth.forms		import UserCreationForm
from django.contrib.auth.decorators import login_required

def registerView(request):
	if request.method=='POST':
		form=UserCreationForm(request.POST)
		if form.is_valid():
			form.save()
			return redirect('login')
	else:
		form=UserCreationForm()	
	return render(request,'registration/register.html',{'form':form})


@login_required
def DashboardView(request):
	return render(request,'pages/Dashboard.html')