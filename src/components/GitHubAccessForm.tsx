import { useEffect, useState } from 'react';
import Link from 'next/link';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { AlertCircle, CheckCircle } from 'lucide-react';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { useForm, ValidationError } from '@formspree/react';

export default function GitHubAccessForm() {
  // Replace "xvgkdrbr" with your actual Formspree form ID
  const [state, handleSubmit] = useForm("xvgkdrbr");
  const [hasSubmitted, setHasSubmitted] = useState(false);
  
  // Check localStorage on component mount to see if the form has been submitted before
  useEffect(() => {
    const submitted = localStorage.getItem('githubAccessFormSubmitted');
    if (submitted === 'true') {
      setHasSubmitted(true);
    }
  }, []);
  
  // Update localStorage when the form is successfully submitted
  useEffect(() => {
    if (state.succeeded) {
      localStorage.setItem('githubAccessFormSubmitted', 'true');
      setHasSubmitted(true);
    }
  }, [state.succeeded]);

  return (
    <Card className="w-full max-w-md mx-auto">
      <CardHeader>
        <CardDescription>
          Enter your GitHub username to request access to the project repositories.
        </CardDescription>
      </CardHeader>
      <CardContent>
        {state.succeeded || hasSubmitted ? (
          <Alert className="bg-green-50 border-green-200">
            <CheckCircle className="h-4 w-4 text-green-600" />
            <AlertTitle>Sent!</AlertTitle>
            <AlertDescription className="text-green-700 mt-1">
        Thank you for your request. You will be added as a collaborator soon.
      </AlertDescription>
          </Alert>
        ) : (
          <form onSubmit={handleSubmit}>
            <div className="grid w-full items-center gap-4">
              <div className="flex flex-col space-y-1.5">
                <Label htmlFor="github-username">GitHub Username</Label>
                <Input
                  id="github-username"
                  name="githubUsername"
                  placeholder="yourusername"
                  required
                  disabled={state.submitting}
                />
                <ValidationError 
                  prefix="GitHub Username" 
                  field="githubUsername"
                  errors={state.errors}
                  className="text-sm text-red-500"
                />
              </div>
            </div>
            
            {state.errors && Object.keys(state.errors).length > 0 && (
              <Alert className="mt-4 bg-red-50 border-red-200">
                <AlertCircle className="h-4 w-4 text-red-600" />
                <AlertTitle>Error</AlertTitle>
                <AlertDescription>
                  There was a problem submitting your request. Please try again.
                </AlertDescription>
              </Alert>
            )}
            
            <div className="mt-4">
              <Button 
                type="submit" 
                className="w-full"
                disabled={state.submitting}
              >
                {state.submitting ? 'Submitting...' : 'Request Access'}
              </Button>
            </div>
          </form>
        )}
        <p className="text-sm text-muted-foreground mt-3">
          If you would like to add another collaborator,{' '}
          <button
            onClick={() => {
              // Find the contact section on the current page
              const contactSection = document.querySelector('section#contact');
              if (contactSection) {
                contactSection.scrollIntoView({ behavior: 'smooth' });
              } else {
                // Fallback to scrolling to bottom if contact section isn't found
                window.scrollTo({
                  top: document.documentElement.scrollHeight,
                  behavior: 'smooth'
                });
              }
            }}
            className="text-primary underline hover:text-primary/80 inline bg-transparent border-0 p-0 cursor-pointer"
          >
            contact me
          </button>.
        </p>
      </CardContent>
    </Card>
  );
}
